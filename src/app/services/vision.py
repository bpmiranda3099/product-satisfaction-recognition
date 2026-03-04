import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from ultralytics import YOLO

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

COCO_KPT_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


@dataclass
class ProductDetection:
    xyxy: Tuple[int, int, int, int]
    conf: float


class VisionService:
    def __init__(self, models_dir: Path, object_model_file: str, pose_model_file: str, object_conf_threshold: float):
        self.object_conf_threshold = object_conf_threshold
        self.object_model = self._load_model(models_dir / object_model_file, "object")
        self.pose_model = YOLO(str(models_dir / pose_model_file))

        self.hands = None
        self.face_mesh = None
        self.mp_draw = None
        self.mp_draw_styles = None
        self.mp_hands = None
        self.mp_face_mesh = None

        if hasattr(mp, "solutions"):
            try:
                self.mp_hands = mp.solutions.hands  # type: ignore[attr-defined]
                self.mp_face_mesh = mp.solutions.face_mesh  # type: ignore[attr-defined]
                self.mp_draw = mp.solutions.drawing_utils  # type: ignore[attr-defined]
                self.mp_draw_styles = mp.solutions.drawing_styles  # type: ignore[attr-defined]

                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=4,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as e:
                print("MediaPipe hands/face mesh failed to initialize; overlays are disabled.")
                print(f"MediaPipe init error details: {e}")

    @staticmethod
    def _load_model(model_path: Path, kind: str):
        try:
            return YOLO(str(model_path))
        except Exception as e:
            print(f"YOLO {kind} initialization failed. {kind.capitalize()} detection disabled.")
            print(f"YOLO error details: {e}")
            return None

    def detect_products(self, frame) -> List[ProductDetection]:
        if self.object_model is None:
            return []

        detections: List[ProductDetection] = []
        results = self.object_model(frame, verbose=False)[0]
        if not hasattr(results, "boxes"):
            return detections

        names = getattr(results, "names", {}) or {}
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, "__iter__") else box.xyxy
            conf = float(box.conf[0]) if hasattr(box.conf, "__iter__") else float(box.conf)
            cls = int(box.cls[0]) if hasattr(box.cls, "__iter__") else int(box.cls)
            if conf < self.object_conf_threshold:
                continue
            class_name = str(names.get(cls, "")).lower()
            if cls == 0 or class_name == "person":
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            detections.append(ProductDetection(xyxy=(x1, y1, x2, y2), conf=conf))
        return detections

    def detect_pose(self, frame, keypoint_conf_threshold: float):
        body_wrist_points = []
        pose_keypoints = []
        if self.pose_model is None:
            return pose_keypoints, body_wrist_points

        pose_results = self.pose_model(frame, verbose=False)[0]
        if hasattr(pose_results, "keypoints") and pose_results.keypoints is not None:
            xy = pose_results.keypoints.xy
            conf = pose_results.keypoints.conf
            if xy is not None:
                xy_np = xy.cpu().numpy()
                conf_np = conf.cpu().numpy() if conf is not None else None
                for pi, person_pts in enumerate(xy_np):
                    person_conf = conf_np[pi] if conf_np is not None else [1.0] * len(person_pts)
                    kp = []
                    for ki, pxy in enumerate(person_pts):
                        px, py = int(pxy[0]), int(pxy[1])
                        pc = float(person_conf[ki]) if ki < len(person_conf) else 1.0
                        kp.append((px, py, pc))
                    pose_keypoints.append(kp)
                    for wrist_idx in (9, 10):
                        if wrist_idx < len(kp):
                            wx, wy, wc = kp[wrist_idx]
                            if wc > keypoint_conf_threshold:
                                body_wrist_points.append((wx, wy))
        return pose_keypoints, body_wrist_points

    def detect_hands_and_face(self, frame, w: int, h: int):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_bboxes = []
        hand_landmarks_list = []
        if self.hands is not None:
            hands_res = self.hands.process(img_rgb)
            if hands_res.multi_hand_landmarks:
                for hand_landmarks in hands_res.multi_hand_landmarks:
                    hand_landmarks_list.append(hand_landmarks)
                    xs = [lm.x for lm in hand_landmarks.landmark]
                    ys = [lm.y for lm in hand_landmarks.landmark]
                    x_min = max(0, int(min(xs) * w) - 12)
                    y_min = max(0, int(min(ys) * h) - 12)
                    x_max = min(w, int(max(xs) * w) + 12)
                    y_max = min(h, int(max(ys) * h) + 12)
                    hand_bboxes.append((x_min, y_min, x_max, y_max))

        return img_rgb, hand_bboxes, hand_landmarks_list

    def detect_face_mesh(self, img_rgb):
        if self.face_mesh is None:
            return None
        return self.face_mesh.process(img_rgb)

    def draw_pose(self, frame, pose_keypoints, keypoint_conf_threshold: float):
        for person_kp in pose_keypoints:
            for a, b in COCO_KPT_CONNECTIONS:
                if a < len(person_kp) and b < len(person_kp):
                    ax, ay, ac = person_kp[a]
                    bx, by, bc = person_kp[b]
                    if ac > keypoint_conf_threshold and bc > keypoint_conf_threshold:
                        cv2.line(frame, (ax, ay), (bx, by), (0, 200, 255), 2)
            for px, py, pc in person_kp:
                if pc > keypoint_conf_threshold:
                    cv2.circle(frame, (px, py), 4, (0, 120, 255), -1)

    def draw_hands(self, frame, hand_landmarks_list, body_wrist_points):
        if self.mp_draw is None or self.mp_hands is None:
            return
        for hand_landmarks in hand_landmarks_list:
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw_styles.get_default_hand_landmarks_style() if self.mp_draw_styles else None,
                self.mp_draw_styles.get_default_hand_connections_style() if self.mp_draw_styles else None,
            )
            wrist = hand_landmarks.landmark[0]
            h, w = frame.shape[:2]
            hx, hy = int(wrist.x * w), int(wrist.y * h)
            if body_wrist_points:
                closest = min(body_wrist_points, key=lambda p: (p[0] - hx) ** 2 + (p[1] - hy) ** 2)
                cv2.line(frame, closest, (hx, hy), (255, 180, 0), 2)

    def draw_face_mesh(self, frame, face_mesh_res):
        if (
            self.mp_draw is None
            or self.mp_face_mesh is None
            or face_mesh_res is None
            or not face_mesh_res.multi_face_landmarks
        ):
            return
        self.mp_draw.draw_landmarks(
            image=frame,
            landmark_list=face_mesh_res.multi_face_landmarks[0],
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_draw_styles.get_default_face_mesh_contours_style() if self.mp_draw_styles else None,
        )

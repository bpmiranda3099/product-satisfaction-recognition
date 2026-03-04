import os
import platform
from pathlib import Path

import cv2

from .config import AppConfig
from .services.capture import VideoCaptureFactory
from .services.reaction import ReactionService
from .services.vision import ProductDetection, VisionService
from .utils.geometry import iou_box

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")


class ProductSatisfactionApp:
    def __init__(self, config: AppConfig, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.models_dir = AppConfig.ensure_models_dir(project_root)

        print("If this is first run models may download (cached automatically).")

        self.vision = VisionService(
            models_dir=self.models_dir,
            object_model_file=config.object_model_file,
            pose_model_file=config.pose_model_file,
            object_conf_threshold=config.object_conf_threshold,
        )
        self.reactions = ReactionService()

    def run(self):
        source = self.config.source
        if isinstance(source, str) and source != "0":
            cap = VideoCaptureFactory.open_capture(source)
        else:
            cap = VideoCaptureFactory.open_capture(0 if str(source) == "0" else int(source))

        if cap is None or not cap.isOpened():
            print("Could not open video/camera source")
            if platform.system() == "Darwin":
                print("On macOS, allow camera access for the app running this terminal:")
                print("System Settings > Privacy & Security > Camera")
                print("Enable access for Terminal / iTerm / Visual Studio Code, then rerun.")
                print("If needed, try a different index: python3 main.py --source 1")
            return

        print("Camera/video started. Press 'q' to quit.")
        window_name = "Product Handling + Expression"
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out = self.process_frame(frame)
            cv2.imshow(window_name, out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        out = frame.copy()

        products = self.vision.detect_products(frame)
        pose_keypoints, body_wrist_points = self.vision.detect_pose(frame, self.config.keypoint_conf_threshold)
        img_rgb, hand_bboxes, hand_landmarks_list = self.vision.detect_hands_and_face(frame, w, h)

        best_handled = self._best_handled_product(products, hand_bboxes)
        is_handling_product = best_handled is not None

        if best_handled is not None:
            best_conf, box = best_handled
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                out,
                f"HANDLED {best_conf:.2f}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

        self.vision.draw_pose(out, pose_keypoints, self.config.keypoint_conf_threshold)
        self.vision.draw_hands(out, hand_landmarks_list, body_wrist_points)

        if is_handling_product:
            face_mesh_res = self.vision.detect_face_mesh(img_rgb)
            reaction = self._infer_reaction(face_mesh_res, w, h)
            if reaction is not None:
                face_box, (label, score) = reaction
                fx1, fy1, fx2, fy2 = face_box
                cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (255, 255, 0), 2)
                cv2.putText(
                    out,
                    f"Reaction: {label} ({score:.2f})",
                    (fx1, fy1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                )
            self.vision.draw_face_mesh(out, face_mesh_res)

        return out

    def _best_handled_product(self, products: list[ProductDetection], hand_bboxes):
        candidates = []
        for product in products:
            for hand_box in hand_bboxes:
                if iou_box(product.xyxy, hand_box) > self.config.handle_iou_threshold:
                    candidates.append((product.conf, product.xyxy))
                    break
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])

    def _infer_reaction(self, face_mesh_res, width: int, height: int):
        if face_mesh_res is None or not face_mesh_res.multi_face_landmarks:
            return None

        face_landmarks = face_mesh_res.multi_face_landmarks[0]
        xs = [lm.x for lm in face_landmarks.landmark]
        ys = [lm.y for lm in face_landmarks.landmark]
        x1 = max(0, int(min(xs) * width) - 10)
        y1 = max(0, int(min(ys) * height) - 10)
        x2 = min(width, int(max(xs) * width) + 10)
        y2 = min(height, int(max(ys) * height) + 10)

        reaction_label, reaction_conf = self.reactions.infer(face_landmarks)
        return (x1, y1, x2, y2), (reaction_label, reaction_conf)

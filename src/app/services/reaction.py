from ..utils.geometry import clamp01


class ReactionService:
    @staticmethod
    def _point(landmarks, idx: int):
        lm = landmarks[idx]
        return lm.x, lm.y

    @staticmethod
    def _dist(a, b) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    def infer(self, face_landmarks):
        lms = face_landmarks.landmark
        eye_scale = self._dist(self._point(lms, 33), self._point(lms, 263))
        if eye_scale <= 1e-6:
            return "considering", 0.0

        mouth_open = self._dist(self._point(lms, 13), self._point(lms, 14)) / eye_scale
        mouth_width = self._dist(self._point(lms, 61), self._point(lms, 291)) / eye_scale
        left_eye_open = self._dist(self._point(lms, 159), self._point(lms, 145)) / eye_scale
        right_eye_open = self._dist(self._point(lms, 386), self._point(lms, 374)) / eye_scale
        eye_open = (left_eye_open + right_eye_open) / 2.0
        mouth_center_y = (self._point(lms, 13)[1] + self._point(lms, 14)[1]) / 2.0
        corners_y = (self._point(lms, 61)[1] + self._point(lms, 291)[1]) / 2.0
        brow_y = (self._point(lms, 70)[1] + self._point(lms, 300)[1]) / 2.0
        upper_eye_y = (self._point(lms, 159)[1] + self._point(lms, 386)[1]) / 2.0
        brow_eye_gap = (upper_eye_y - brow_y) / eye_scale
        corner_offset = corners_y - mouth_center_y
        brow_inner_dist = self._dist(self._point(lms, 105), self._point(lms, 334)) / eye_scale

        scores = {
            "positive": clamp01(
                max(mouth_width - 0.53, 0) * 8
                + max(0.008 - corner_offset, 0) * 30
                + max(0.03 - mouth_open, 0) * 4
            ),
            "dissatisfied": clamp01(
                max(corner_offset - 0.001, 0) * 34
                + max(0.060 - mouth_open, 0) * 8
                + max(0.52 - mouth_width, 0) * 6
            ),
            "curious": clamp01(max(brow_eye_gap - 0.175, 0) * 12 + max(0.035 - mouth_open, 0) * 8),
            "considering": clamp01(max(eye_open - 0.017, 0) * 12 + max(0.060 - mouth_open, 0) * 8),
            "frustrated": clamp01(
                max(0.50 - brow_inner_dist, 0) * 26
                + max(corner_offset - 0.002, 0) * 16
                + max(0.13 - brow_eye_gap, 0) * 12
            ),
        }

        if brow_inner_dist < 0.46 and corner_offset > -0.002:
            scores["frustrated"] = clamp01(scores["frustrated"] + 0.12)
            scores["dissatisfied"] = clamp01(scores["dissatisfied"] + 0.05)

        label = max(scores, key=scores.get)
        return label, scores[label]

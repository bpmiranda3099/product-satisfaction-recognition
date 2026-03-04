from typing import Tuple

Box = Tuple[int, int, int, int]


def iou_box(a: Box, b: Box) -> float:
    x_a = max(a[0], b[0])
    y_a = max(a[1], b[1])
    x_b = min(a[2], b[2])
    y_b = min(a[3], b[3])
    inter_w = max(0, x_b - x_a)
    inter_h = max(0, y_b - y_a)
    inter = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

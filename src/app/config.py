from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    source: str = "0"
    object_conf_threshold: float = 0.25
    keypoint_conf_threshold: float = 0.2
    handle_iou_threshold: float = 0.02
    object_model_file: str = "yolov8n.pt"
    pose_model_file: str = "yolo26n-pose.pt"

    @staticmethod
    def project_root(main_file: str) -> Path:
        return Path(main_file).resolve().parent

    @staticmethod
    def ensure_models_dir(project_root: Path) -> Path:
        models_dir = project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

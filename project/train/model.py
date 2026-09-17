from typing import Any
from ultralytics import YOLO
from trainer import MinIOTrainer


class MinioYOLO(YOLO):
    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        mapping = super().task_map
        mapping["detect"]["trainer"] = MinIOTrainer
        return mapping

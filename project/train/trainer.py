from typing import Any, Literal
from ultralytics.data import BaseDataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.utils import colorstr

from minio_dataset import MinIODataset, parse_custom_scheme


class MinIOTrainer(DetectionTrainer):
    def get_dataset(self) -> dict[str, Any]:
        self.data_cfg = parse_custom_scheme(self.args.data)
        self.data_cfg.setdefault("channels", 3)
        return self.data_cfg

    def build_dataset(
        self,
        img_path: str,
        mode: Literal['train', 'val'] = "train",
        batch: int | None = None
    ) -> BaseDataset:
        fraction = self.args.fraction if mode == "train" else 1.0
        pad = 0.0 if mode == "train" else 0.5
        split = self.data_cfg[mode]
        gs = max(int(unwrap_model(self.model).stride.max()), 32)

        return MinIODataset(
            img_path="",
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect if mode == "train" else True,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls,
            stride=gs,
            pad=pad,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=fraction,
            bucket=self.data_cfg["bucket"],
            img_prefixes=split["images"],
            label_prefixes=split["labels"],
        )

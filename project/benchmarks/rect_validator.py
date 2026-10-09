import cv2
import numpy as np
import torch
from unittest.mock import patch
from typing import Any, Literal

from project.train.minio_dataset import MinIODataset, parse_custom_scheme
from project.benchmarks.test.rect_dataset import RectValMixin

from ultralytics.data import BaseDataset
import ultralytics.engine.validator as validator_module
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, colorstr, nms
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.utils.checks import check_imgsz

PERSON_ALIASES = {"person", "persons", "people", "human", "humans", "pedestrian", "pedestrians"}


def find_person_id(names: dict[str, Any] | list[str]) -> int:
    names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else names
    for k, v in names.items():
        if str(v).strip().lower() in PERSON_ALIASES:
            return int(k)

    if len(names) == 1:
        return int(next(iter(names)))

    raise ValueError(f"No 'person' class in names: {names}")


class RectMinIODataset(RectValMixin, MinIODataset):
    def _read_raw(self, i: int) -> np.ndarray:
        file = self.im_files[i]
        try:
            data = self._get_bytes(file)
        except Exception as error:
            LOGGER.error(error)
            raise FileNotFoundError(f"Failed to download '{file}' from 'minio://{self.bucket}'")

        im = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"Failed to decode '{file}' from 'minio://{self.bucket}'")
        return im


class RectValidator(DetectionValidator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target_hw: tuple[int, int] | None = None
        self.val_mode: Literal['letterbox', 'mosaic', 'collage'] | None = None
        self.grid: int = 2  # collage grid: 2 -> 2x2, 3 -> 3x3
        self.person_id: int = 0  # person index in the MODEL's classes, set in init_metrics

    def __call__(self, trainer=None, model=None, **kwargs):
        with patch.object(validator_module, "check_det_dataset", lambda data, **kw: self.get_dataset()):
            return super().__call__(trainer, model, **kwargs)

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self.person_id = find_person_id(model.names)
        n_model_classes = len(model.names)
        if self.end2end and n_model_classes > 1:
            # end-to-end heads (yolo26, yolov10) take top-k over ALL classes inside the model;
            # widen it so that max_det person boxes survive, the cut to max_det happens in postprocess
            native = model.model if getattr(model, "format", None) == "pt" else model
            if hasattr(native, "set_head_attr"):
                native.set_head_attr(max_det=self.args.max_det * min(n_model_classes, 10))
        LOGGER.info(
            f"{colorstr('val:')} person-only, model class {self.person_id} "
            f"('{model.names[self.person_id]}' of {n_model_classes})"
        )
        self.names = {0: "person"}
        self.nc = 1
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(names=self.names, save_matches=self.args.plots and self.args.visualize)

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        outputs = nms.non_max_suppression(
            preds, self.args.conf, self.args.iou,
            classes=[self.person_id], multi_label=True,
            agnostic=self.args.agnostic_nms, max_det=self.args.max_det, end2end=self.end2end,
        )
        for x in outputs:
            x[:, 5] = 0
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def plot_predictions(self, batch, *args, **kwargs) -> None:
        return super().plot_predictions({**batch, "img": batch["img"].clone()}, *args, **kwargs)

    def get_dataset(self) -> dict[str, Any]:
        self.data_cfg = parse_custom_scheme(self.args.data)
        self.data_cfg.setdefault("channels", 3)
        return self.data_cfg

    def build_dataset(
        self,
        img_path: str,
        mode: Literal["train", "val"] = "val",
        batch: int | None = None
    ) -> BaseDataset:
        self.data_cfg = parse_custom_scheme(self.args.data)
        self.data_cfg.setdefault("channels", 3)
        split = self.data_cfg[mode]
        gs = max(int(self.stride.max() if hasattr(self.stride, "max") else self.stride), 32)

        requested = self.target_hw or self.args.imgsz
        self.target_hw = tuple(check_imgsz(requested, stride=gs, min_dim=2, max_dim=2))
        val_mode = self.val_mode or ("mosaic" if self.args.mosaic > 0 else "letterbox")
        extra = f" {self.grid}x{self.grid}, cell {self.target_hw[0] // self.grid}x{self.target_hw[1] // self.grid}" if val_mode == "collage" else ""
        LOGGER.info(f"{colorstr('val:')} input {self.target_hw[0]}x{self.target_hw[1]} (HxW), mode={val_mode}{extra}")

        return RectMinIODataset(
            img_path="",
            target_hw=self.target_hw,
            val_mode=val_mode,
            grid=self.grid,
            scaleup=True,  # like stock val: small images are upscaled to the input size
            seed=self.args.seed,
            keep_class=find_person_id(self.data_cfg["names"]),
            batch_size=batch,
            hyp=self.args,
            cache=None,
            single_cls=False,
            stride=gs,
            pad=0.0,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=None,  # class filtering is done by keep_class
            data=self.data,
            fraction=1.0,
            bucket=self.data_cfg["bucket"],
            img_prefixes=split["images"],
            label_prefixes=split["labels"],
        )

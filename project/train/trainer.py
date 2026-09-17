from typing import Any, Literal
from ultralytics.data import BaseDataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER, RANK, colorstr

from minio_dataset import MinIODataset, MinIODetectionValidator, parse_custom_scheme
from mlflow_extend import attach_extra_mlflow_callbacks


class MinIOTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        attach_extra_mlflow_callbacks(self)

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

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return MinIODetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=self.args.__class__(**vars(self.args)),
            _callbacks=self.callbacks,
        )

    def final_eval(self) -> None:
        if not self.best.exists():
            return

        LOGGER.info(f"\nValidating {self.best}...")
        model, _ = load_checkpoint(str(self.best), device=self.device)
        model = model.float().to(self.device)

        self.validator.args.plots = self.args.plots
        self.validator.args.compile = False
        self.metrics = self.validator(trainer=self, model=model)

        # In the trainer-context branch of BaseValidator.__call__ (self.training=True, which we
        # force here to avoid check_det_dataset), non-master DDP ranks (RANK > 0) hit a bare
        # `return` and get back None instead of a stats dict — unlike the standalone branch this
        # override bypasses, which returns {} on non-master ranks. Only rank -1 (single-GPU) or 0
        # (DDP master) get real metrics; skip the rest on other ranks.
        if RANK not in {-1, 0} or self.metrics is None:
            return

        self.metrics.pop("fitness", None)
        self.epoch += 1  # log best metrics at step epochs+1, not overwriting last epoch
        self.run_callbacks("on_fit_epoch_end")
        self.epoch -= 1

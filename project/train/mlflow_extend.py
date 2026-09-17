import os
import shutil
from pathlib import Path
import yaml
import mlflow
import tempfile
from ultralytics.utils import LOGGER, RANK, colorstr

PREFIX = colorstr("MLflow: ")


# Filenames DetectionValidator (over)writes every validation call when args.plots=True.
_VAL_PLOT_GLOBS = (
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PR_curve.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
    "val_batch*_labels.jpg",
    "val_batch*_pred.jpg",
)

# Set to True to delete the local copy of each file right after it's been uploaded to MLflow for
# that epoch, so per-epoch validation images don't keep accumulating on local disk. Off by default
# because DetectionValidator overwrites these same filenames on the very next epoch anyway (no
# unbounded growth), and keeping the last epoch's copy locally is convenient for quick inspection
# without opening MLflow. Flip via env var if disk space is a real constraint.
DELETE_LOCAL_AFTER_UPLOAD = os.getenv("MLFLOW_DELETE_LOCAL_VAL_IMAGES", "false").lower() in (
    "1", "true", "yes", "on", "y", "t",
)

# Name under which the best checkpoint is registered in the MLflow Model Registry. Override via
# env var if you want per-project/per-task registered model names instead of one shared name.
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "yolo-minio")


def force_val_plots_every_epoch(validator) -> None:
    validator.args.plots = True


def log_run_config(trainer) -> None:
    if not mlflow or not getattr(trainer, "_mlflow_active", False):
        return
    if RANK not in (-1, 0):
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        try:
            args_path = tmp / "train_args.yaml"
            args_path.write_text(yaml.safe_dump(dict(trainer.args), sort_keys=False, default_flow_style=False))
            mlflow.log_artifact(str(args_path), artifact_path="config")
        except Exception as e:
            LOGGER.warning(f"{PREFIX}failed to log train_args.yaml: {e}")

        raw_data = getattr(trainer.args, "data", None)
        if isinstance(raw_data, (str, os.PathLike)) and Path(raw_data).suffix in (".yaml", ".yml"):
            src = Path(raw_data)
            if src.exists():
                try:
                    dst = tmp / "dataset.yaml"
                    shutil.copyfile(src, dst)
                    mlflow.log_artifact(str(dst), artifact_path="config")
                except Exception as e:
                    LOGGER.warning(f"{PREFIX}failed to log dataset.yaml: {e}")

        data_cfg = getattr(trainer, "data_cfg", None)
        if isinstance(data_cfg, dict):
            try:
                resolved_path = tmp / "dataset_resolved.yaml"
                resolved_path.write_text(yaml.safe_dump(data_cfg, sort_keys=False, default_flow_style=False))
                mlflow.log_artifact(str(resolved_path), artifact_path="config")
            except Exception as e:
                LOGGER.warning(f"{PREFIX}failed to log dataset_resolved.yaml: {e}")


def log_epoch_validation_artifacts(trainer) -> None:
    if not mlflow or not getattr(trainer, "_mlflow_active", False):
        return
    if RANK not in (-1, 0):
        return

    save_dir = Path(trainer.save_dir)
    epoch = trainer.epoch + 1  # trainer.epoch is 0-indexed during training
    artifact_path = f"val_images/epoch_{epoch:04d}"

    found_any = False
    for pattern in _VAL_PLOT_GLOBS:
        for f in save_dir.glob(pattern):
            try:
                mlflow.log_artifact(str(f), artifact_path=artifact_path)
                found_any = True
                if DELETE_LOCAL_AFTER_UPLOAD:
                    f.unlink(missing_ok=True)
            except Exception as e:
                LOGGER.warning(f"{PREFIX}failed to log '{f.name}' for epoch {epoch}: {e}")

    if found_any:
        LOGGER.debug(f"{PREFIX}logged epoch {epoch} validation images to '{artifact_path}'")


def register_best_model(trainer) -> None:
    if not mlflow or not getattr(trainer, "_mlflow_active", False):
        return
    if RANK not in (-1, 0):
        return
    if not trainer.best.exists():
        return

    active_run = mlflow.active_run()
    if active_run is None:
        LOGGER.warning(f"{PREFIX}no active run, skipping model registration")
        return

    run_id = active_run.info.run_id
    model_uri = f"runs:/{run_id}/{trainer.best.parent.name}/{trainer.best.name}"
    try:
        result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
        LOGGER.info(
            f"{PREFIX}registered '{REGISTERED_MODEL_NAME}' version {result.version} "
            f"from {model_uri}"
        )
    except Exception as e:
        LOGGER.warning(f"{PREFIX}failed to register model: {e}")


def attach_extra_mlflow_callbacks(trainer) -> None:
    if not mlflow:
        return
    trainer.add_callback("on_pretrain_routine_end", log_run_config)
    trainer.add_callback("on_val_start", force_val_plots_every_epoch)
    trainer.add_callback("on_fit_epoch_end", log_epoch_validation_artifacts)
    trainer.add_callback("on_train_end", register_best_model)

import logging
import mlflow
from dotenv import load_dotenv

load_dotenv(".env")

from ultralytics import settings, YOLO
from trainer import MinIOTrainer


def main(model_name: str) -> None:
    model = YOLO(f"{model_name}.pt")
    model.train(
        data="custom.yaml",
        trainer=MinIOTrainer,
        epochs=100,
        imgsz=640,
        workers=4,
        cache=False,
        batch=64
    )


if __name__ == "__main__":
    model_name = "yolov8n"

    logging.getLogger("ultralytics").setLevel(logging.DEBUG)
    settings.update(
        {
            "mlflow": True,
            "runs_dir": "./runs",
            "weights_dir": "./weights",
            "tensorboard": False
        }
    )

    mlflow.set_experiment(f"{model_name}-s3")
    with mlflow.start_run():
      main(model_name)

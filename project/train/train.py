import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
import os
import sys
from pathlib import Path

THIS_DIR = str(Path(__file__).resolve().parent)

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

os.environ["PYTHONPATH"] = os.pathsep.join(
    filter(None, [THIS_DIR, os.environ.get("PYTHONPATH", "")])
)

import logging
import mlflow
from model import MinioYOLO
from dotenv import load_dotenv

load_dotenv(".env")

from ultralytics import settings


def main(model_name: str) -> None:
    model = MinioYOLO(f"{model_name}.pt")
    model.train(
        device="0,1",
        data="custom.yaml",
        epochs=100,
        imgsz=640,
        workers=4,
        cache=False,
        batch=64,
        plots = True
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

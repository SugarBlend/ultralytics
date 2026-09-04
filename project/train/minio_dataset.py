import math
import threading
from pathlib import Path
from urllib.parse import urlparse
import os
import cv2
import numpy as np
import yaml
from minio import Minio

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER

from urllib3.exceptions import MaxRetryError, ProtocolError
from minio.error import S3Error
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from typing import List, Dict, Any
from pydantic import BaseModel, Field, model_validator


def parse_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


class ImagesLabels(BaseModel):
    images: List[str] = Field()
    labels: List[str] = Field()

    @model_validator(mode="after")
    def check_same_length(self) -> "ImagesLabels":
        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Number elements of images ({len(self.images)}) and labels "
                f"({len(self.labels)}) must be the same."
            )
        return self


class LabelModel(BaseModel):
    path: str = Field()
    train: ImagesLabels = Field()
    val: ImagesLabels = Field()
    names: Dict[int, str] = Field()


def parse_custom_scheme(yaml_path: str) -> dict[str, Any]:
    yaml_path = Path(yaml_path)
    with yaml_path.open("r") as file:
        dataset_config = LabelModel.model_validate(yaml.safe_load(file))

    root = dataset_config.path.rstrip("/")
    bucket, root_prefix = parse_uri(root)

    def resolve(split: str) -> dict[str, list[str]]:
        images = getattr(dataset_config, split).images
        labels = getattr(dataset_config, split).labels

        return {
            "images": [f"{root_prefix}/{path}".strip("/") for path in images],
            "labels": [f"{root_prefix}/{path}".strip("/") for path in labels]
        }

    return {
        "bucket": bucket,
        "train": resolve("train"),
        "val": resolve("val"),
        "names": dataset_config.names,
        "nc": len(dataset_config.names),
    }


class MinIODataset(YOLODataset):
    _img_extensions: list[str] = tuple((os.getenv("IMG_EXTENSIONS") or ".jpg,.jpeg,.png,.bmp").split(","))
    _minio_endpoint: str = os.getenv("S3_ENDPOINT")
    _minio_access_key: str = os.getenv("S3_ACCESS_KEY")
    _minio_secret_key: str = os.getenv("S3_SECRET_KEY")
    _minio_secure = False
    _thread_local = threading.local()

    def __init__(
        self,
        *args,
        bucket: str,
        img_prefixes: list[str],
        label_prefixes: list[str],
        **kwargs,
    ):
        self.bucket: str = bucket
        self.img_prefixes: list[str] = [p.rstrip("/") + "/" for p in img_prefixes]
        self.label_prefixes: list[str] = [p.rstrip("/") + "/" for p in label_prefixes]
        super().__init__(*args, **kwargs)

    @classmethod
    def _client(cls) -> Minio:
        if not hasattr(cls._thread_local, "client"):
            cls._thread_local.client = Minio(
                cls._minio_endpoint,
                access_key=cls._minio_access_key,
                secret_key=cls._minio_secret_key,
                secure=cls._minio_secure,
            )
        return cls._thread_local.client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type((S3Error, ProtocolError, MaxRetryError, ConnectionError, OSError)),
        before_sleep=before_sleep_log(LOGGER, logging.WARNING),
        reraise=True,
    )
    def _get_bytes(self, key: str) -> bytes:
        response = None
        try:
            response = self._client().get_object(self.bucket, key)
            return response.read()
        finally:
            if response:
                response.close()
                response.release_conn()

    def get_img_files(self, img_path: str) -> list[str]:
        client = self._client()
        files: list[str] = []

        for img_prefix, label_prefix in zip(self.img_prefixes, self.label_prefixes):
            img_objs = sorted(
                o.object_name for o in client.list_objects(self.bucket, prefix=img_prefix, recursive=True)
                if o.object_name.lower().endswith(self._img_extensions)
            )

            label_stems = {
                Path(o.object_name).stem
                for o in client.list_objects(self.bucket, prefix=label_prefix, recursive=True)
                if o.object_name.lower().endswith(".txt")
            }

            kept = [f for f in img_objs if Path(f).stem in label_stems]
            skipped = len(img_objs) - len(kept)

            LOGGER.info(
                f"MinIODataset: found {len(img_objs)} images in {img_prefix}, "
                f"{len(kept)} with labels, {skipped} skipped (no annotation)"
            )
            files.extend(kept)

        if not files:
            raise FileNotFoundError(
                f"Doesn't found any images with labels in 'minio://{self.bucket}/{self.img_prefixes}'")

        return files

    def _label_prefix_for(self, img_key: str) -> tuple[str, str]:
        for img_prefix, label_prefix in zip(self.img_prefixes, self.label_prefixes):
            if img_key.startswith(img_prefix):
                return img_prefix, label_prefix
        raise ValueError(f"Doesn't found correspondence label-prefix for {img_key}")

    def img2label_paths(self, img_paths: list[str]) -> list[str]:
        result: list[str] = []
        for path in img_paths:
            img_prefix, label_prefix = self._label_prefix_for(path)
            rel = Path(path[len(img_prefix):]).with_suffix(".txt").as_posix()
            result.append(label_prefix + rel)
        return result

    def _ensure_label_loaded(self, index: int) -> dict[str, Any]:
        label = self.labels[index]
        if label.get("_lb_loaded"):
            return label

        try:
            raw = self._get_bytes(self.label_files[index]).decode("utf-8")
        except Exception as error:
            LOGGER.error(f"Failed to load label for {label['im_file']}: {error}")
            label["_lb_loaded"] = True
            return label

        lb: list[list[float]] = []
        for line in raw.strip().splitlines():
            parts = line.split()
            if parts:
                lb.append(list(map(float, parts)))

        if lb:
            lb_arr: np.ndarray = np.array(lb, dtype=np.float32)
            label["cls"] = lb_arr[:, :1]
            label["bboxes"] = lb_arr[:, 1:]

        label["_lb_loaded"] = True
        return label

    def get_image_and_label(self, index: int):
        self._ensure_label_loaded(index)
        return super().get_image_and_label(index)

    def load_image(self, i: int, rect_mode: bool = True, resize_short: bool = True):
        img, file, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if img is not None:
            return img, self.im_hw0[i], self.im_hw[i]

        try:
            data = self._get_bytes(file)
        except Exception as error:
            LOGGER.error(error)
            raise FileNotFoundError(f"Failed to download '{file}' from 'minio://{self.bucket}'")

        buf = np.frombuffer(data, dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"Failed to decode '{file}' from 'minio://{self.bucket}'")

        h0, w0 = im.shape[:2]
        if rect_mode:
            if resize_short:  # resize short side to imgsz while maintaining aspect ratio
                r = self.imgsz / min(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (math.ceil(w0 * r), self.imgsz) if h0 < w0 else (self.imgsz, math.ceil(h0 * r))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
            self.buffer.append(i)
            if len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                if self.cache != "ram":
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

        return im, (h0, w0), im.shape[:2]

    def get_labels(self) -> list[dict[str, Any]]:
        self.label_files = self.img2label_paths(self.im_files)

        labels: list[dict[str, Any]] = []
        for im_file in self.im_files:
            labels.append(
                {
                    "im_file": im_file,
                    "shape": (0, 0),
                    "cls": np.zeros((0, 1), dtype=np.float32),
                    "bboxes": np.zeros((0, 4), dtype=np.float32),
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "_lb_loaded": False,
                }
            )
        return labels

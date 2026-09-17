import io
import math
import threading
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
import os
import cv2
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import yaml
from minio import Minio
from minio.error import S3Error
import hashlib
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from typing import List, Dict, Any
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER
from ultralytics.models.yolo.detect import DetectionValidator
from urllib.parse import urlparse
import urllib3
from urllib3.exceptions import MaxRetryError, ProtocolError
import logging
import imagesize


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
    _minio_endpoint: str = urlparse(os.getenv("AWS_ENDPOINT_URL", "")).netloc
    _minio_access_key: str = os.getenv("AWS_ACCESS_KEY_ID")
    _minio_secret_key: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    _minio_secure = False
    _thread_local = threading.local()

    # Byte budgets tried in sequence via range-requests to sniff image dimensions
    # from the file header. Most JPEG/PNG headers fit in the first tier; some
    # JPEGs (large EXIF/ICC blocks, progressive encoding) need more before the
    # SOF marker appears, so we escalate instead of immediately falling back.
    _SHAPE_PROBE_TIERS: tuple[int, ...] = (8192, 65536, 262144)
    # Parallelism used when resolving shapes for all images during get_labels().
    _SHAPE_PROBE_WORKERS: int = 64

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
        pid = os.getpid()
        if getattr(cls._thread_local, "pid", None) != pid:
            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=10, read=60),
                maxsize=32,
                retries=urllib3.Retry(
                    total=3,
                    backoff_factor=0.3,
                    status_forcelist=[500, 502, 503, 504],
                ),
            )
            cls._thread_local.client = Minio(
                cls._minio_endpoint,
                access_key=cls._minio_access_key,
                secret_key=cls._minio_secret_key,
                secure=cls._minio_secure,
                http_client=http_client,
            )
            cls._thread_local.pid = pid
        return cls._thread_local.client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type((S3Error, ProtocolError, MaxRetryError, ConnectionError, OSError)),
        before_sleep=before_sleep_log(LOGGER, logging.WARNING),
        reraise=True,
    )
    def _get_bytes(self, key: str, length: int = 0) -> bytes:
        response = None
        try:
            response = self._client().get_object(self.bucket, key, length=length)
            return response.read()
        finally:
            if response:
                response.close()
                response.release_conn()

    def _probe_image_shape(self, img_key: str) -> tuple[int, int]:
        # Try to fetch image size from first bytes information, need further for rectangular training/validation
        last_error: Exception | None = None
        for n_bytes in self._SHAPE_PROBE_TIERS:
            try:
                header = self._get_bytes(img_key, n_bytes)
                w, h = imagesize.get(io.BytesIO(header))
                if w and h and w > 0 and h > 0:
                    return int(h), int(w)
            except Exception as error:
                last_error = error

        # Last resort: full download and decode. Rare in practice, but avoids
        # silently mis-sizing a real image just because its header is unusual.
        try:
            data = self._get_bytes(img_key)
            buf = np.frombuffer(data, dtype=np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if im is not None:
                h, w = im.shape[:2]
                return int(h), int(w)
        except Exception as error:
            last_error = error

        LOGGER.warning(
            f"{self.prefix}Could not determine dimensions for '{img_key}'"
            f"{f': {last_error}' if last_error else ''}, using fallback"
        )
        return self.imgsz, self.imgsz

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
        # Get bucket prefixes for current relative path
        for img_prefix, label_prefix in zip(self.img_prefixes, self.label_prefixes):
            if img_key.startswith(img_prefix):
                return img_prefix, label_prefix
        raise ValueError(f"Doesn't found correspondence label-prefix for {img_key}")

    def _label_key_for(self, img_key: str) -> str:
        # Return path to txt label file which correspondence with current image
        img_prefix, label_prefix = self._label_prefix_for(img_key)
        rel = Path(img_key[len(img_prefix):]).with_suffix(".txt").as_posix()
        return label_prefix + rel

    def img2label_paths(self, img_paths: list[str]) -> list[str]:
        # Do NOT rely on positional alignment between this list
        # and self.im_files / self.labels elsewhere, since BaseDataset.set_rectangle()
        # (triggered by rect=True) reorders self.im_files and self.labels by aspect ratio
        # but has no knowledge of this separate list, silently desynchronizing it.
        # Label loading always derives the key fresh from label["im_file"] via _label_key_for().
        return [self._label_key_for(p) for p in img_paths]

    def _ensure_label_loaded(self, index: int) -> dict[str, Any]:
        label = self.labels[index]
        if label.get("_lb_loaded"):
            return label

        # Always derive the label key from this label's own im_file, never from
        # a positionally-indexed side list — self.labels/self.im_files may have
        # been reordered (e.g. by set_rectangle() under rect=True) independently
        # of any other list built at __init__ time.
        im_file = label["im_file"]
        label_key = self._label_key_for(im_file)

        try:
            raw = self._get_bytes(label_key).decode("utf-8")
        except Exception as error:
            LOGGER.error(f"Failed to load label for {im_file}: {error}")
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
        if im.ndim == 2:
            im = im[..., None]

        # Add to buffer if training with augmentations
        if self.augment and self.cache != "ram":
            self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                j = self.buffer.pop(0)
                if self.cache != "ram":
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

        return im, (h0, w0), im.shape[:2]

    def get_labels(self) -> list[dict[str, Any]]:
        # Kept for debugging / external inspection only — not used for label
        # loading itself (see _ensure_label_loaded / _label_key_for).
        self.label_files = self.img2label_paths(self.im_files)

        # Resolve real (h, w) for every image up front, in parallel, via cheap
        # range-requests against the object header. This is required for
        # rect=True: BaseDataset.set_rectangle() sorts images by aspect ratio
        # and computes batch_shapes from label["shape"]; a placeholder (0, 0)
        # produces NaN aspect ratios and corrupts both the sort order and the
        # letterbox target size used later in load_image/augmentations.
        LOGGER.info(f"{self.prefix}Resolving image shapes for {len(self.im_files)} images (range-requests)...")

        cache_key = hashlib.md5("".join(self.im_files).encode()).hexdigest()
        cache_path = Path(f"/tmp/shapes_cache_{cache_key}.json")

        if cache_path.exists():
            shapes = json.loads(cache_path.read_text())
            shapes = [tuple(s) for s in shapes]
        else:
            with ThreadPoolExecutor(max_workers=self._SHAPE_PROBE_WORKERS) as ex:
                shapes = list(ex.map(self._probe_image_shape, self.im_files))
            cache_path.write_text(json.dumps(shapes))

        labels: list[dict[str, Any]] = []
        for im_file, (h, w) in zip(self.im_files, shapes):
            labels.append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
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


class MinIODetectionValidator(DetectionValidator):
    def get_model(self, model, trainer=None):
        if model is not None:
            return model
        return super().get_model(model, trainer)

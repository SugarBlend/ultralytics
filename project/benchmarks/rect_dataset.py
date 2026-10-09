import cv2
import math
import random
import numpy as np
from typing import Any, Literal

from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.utils.instance import Instances


class RectValMixin(object):
    def __init__(
        self,
        *args,
        target_hw: tuple[int, int] = (640, 640),
        val_mode: Literal['letterbox', 'mosaic', 'collage'] = "letterbox",
        scaleup: bool = True,
        seed: int = 0,
        mosaic_area_thr: float = 0.1,
        keep_class: int | None = None,
        grid: int = 2,
        collage_shuffle: bool = True,
        **kwargs,
    ):
        assert val_mode in {"letterbox", "mosaic", "collage"}, f"unknown val_mode={val_mode}"
        self.target_hw: tuple[int, int] = target_hw
        self.grid = int(grid)
        self.collage_shuffle = collage_shuffle

        self._collage_order = None
        if val_mode == "collage":
            th, tw = self.target_hw
            assert self.grid >= 2, "collage needs grid >= 2"
            self.cell_hw = (th // self.grid, tw // self.grid)  # remainder (if any) stays gray padding
            self._fit_hw = self.cell_hw  # images are fitted into one cell
        else:
            self._fit_hw = self.target_hw

        self.val_mode = val_mode
        self.scaleup = scaleup
        self.seed = seed
        self.mosaic_area_thr = mosaic_area_thr
        # keep only this GT class and renumber it to 0 (e.g. person-only evaluation).
        # Done per sample, because labels may be loaded lazily (MinIO) and the stock
        # update_labels(include_class) at init would see empty labels.
        self.keep_class = keep_class
        kwargs["imgsz"] = max(self.target_hw)
        kwargs["augment"] = False  # no train augmentations, no image buffer
        kwargs["rect"] = False
        super().__init__(*args, **kwargs)

    def update_labels(self, include_class: list[int] | None) -> None:
        # With keep_class the stock filter must not run: DetectionValidator.get_dataloader calls
        # update_labels(range(len(model.names))) for a 1-class model, which would keep GT class 0
        # instead of the dataset's person class.
        if getattr(self, "keep_class", None) is not None:
            return

        super().update_labels(include_class)

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        label = super().get_image_and_label(index)

        if self.keep_class is not None:
            keep = label["cls"].reshape(-1) == self.keep_class
            label["instances"] = label["instances"][keep]
            label["cls"] = np.zeros((int(keep.sum()), 1), dtype=np.float32)
        return label

    def _read_raw(self, i: int) -> np.ndarray:
        im = cv2.imread(self.im_files[i])
        if im is None:
            raise FileNotFoundError(f"Image not found {self.im_files[i]}")

        return im

    def load_image(self, i: int, *args, **kwargs) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        im = self._read_raw(i)
        h, w, c = im.shape
        th, tw = self._fit_hw
        r = min(th / h, tw / w)

        if not self.scaleup:
            r = min(r, 1.0)

        if r != 1:
            im = cv2.resize(
                im, dsize=(min(round(w * r), tw), min(round(h * r), th)),
                interpolation=cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            )

        return im, (h, w), im.shape[:2]

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        if self.val_mode == "letterbox":
            transforms = Compose([LetterBox(new_shape=self.target_hw, auto=False, scaleup=False, center=True)])
        else:
            transforms = Compose([])  # _rect_mosaic / _collage already return an H x W image

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=0.0,
            )
        )
        return transforms

    def __len__(self) -> int:
        samples = len(self.labels)

        if self.val_mode == "collage":
            return math.ceil(samples / self.grid**2)  # one item - one collage of grid*grid images

        return samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.val_mode == "mosaic":
            func = self._rect_mosaic
        elif self.val_mode == 'collage':
            func = self._collage
        else:
            func = self.get_image_and_label

        return self.transforms(func(index))

    def _collage(self, index: int) -> dict[str, Any]:
        n_img = len(self.labels)

        if self._collage_order is None:  # same seed -> same order in every worker and every run
            order = np.arange(n_img)
            if self.collage_shuffle:  # mix sources, avoid 4 near-identical consecutive video frames
                order = np.random.default_rng(self.seed).permutation(n_img)
            self._collage_order = order

        g = self.grid
        idxs = self._collage_order[index * g * g : (index + 1) * g * g]
        th, tw = self.target_hw
        ch, cw = self.cell_hw

        canvas = None
        first = None
        cls_list, inst_list = [], []
        for c, j in enumerate(idxs):
            lab = self.get_image_and_label(int(j))
            img = lab["img"]
            h, w = img.shape[:2]
            if canvas is None:
                first = lab
                canvas = np.full((th, tw, img.shape[2]), 114, dtype=np.uint8)
            row, col = divmod(c, g)
            ox = col * cw + (cw - w) // 2  # centered inside the cell
            oy = row * ch + (ch - h) // 2
            canvas[oy : oy + h, ox : ox + w] = img

            inst: Instances = lab["instances"]
            inst.convert_bbox(format="xyxy")
            inst.denormalize(w, h)
            inst.add_padding(ox, oy)
            cls_list.append(lab["cls"])
            inst_list.append(inst)

        return {
            "im_file": first["im_file"],
            "img": canvas,
            "ori_shape": (th, tw),  # a collage has no single original image: identity mapping
            "resized_shape": (th, tw),
            "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
            "cls": np.concatenate(cls_list, 0),
            "instances": Instances.concatenate(inst_list, axis=0),
        }

    def _rect_mosaic(self, index: int) -> dict[str, Any]:
        th, tw = self.target_hw
        rng = random.Random(self.seed * 1_000_003 + index)
        indices = [index] + [rng.randrange(len(self)) for _ in range(3)]
        # mosaic center, same range as Ultralytics: [s/2, 3s/2] on a 2s canvas, per axis
        yc = int(rng.uniform(th / 2, 3 * th / 2))
        xc = int(rng.uniform(tw / 2, 3 * tw / 2))

        first = None
        canvas = None
        cls_list, inst_list = [], []
        for k, j in enumerate(indices):
            lab = self.get_image_and_label(j)
            img = lab["img"]
            h, w = img.shape[:2]
            if canvas is None:
                first = lab
                canvas = np.full((2 * th, 2 * tw, img.shape[2]), 114, dtype=np.uint8)
            if k == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif k == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * tw), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif k == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * th, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * tw), min(2 * th, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            inst: Instances = lab["instances"]
            inst.convert_bbox(format="xyxy")
            inst.denormalize(w, h)
            inst.add_padding(x1a - x1b, y1a - y1b)
            inst.clip(2 * tw, 2 * th)  # part of the image that didn't fit on the canvas
            cls_list.append(lab["cls"])
            inst_list.append(inst)

        instances = Instances.concatenate(inst_list, axis=0)
        cls = np.concatenate(cls_list, 0)

        # crop the central H x W window
        oy, ox = th // 2, tw // 2
        img = np.ascontiguousarray(canvas[oy : oy + th, ox : ox + tw])
        instances.add_padding(-ox, -oy)
        before = instances.bboxes.copy()
        instances.clip(tw, th)
        after = instances.bboxes

        # drop boxes that were cut almost entirely (same rule as Ultralytics box_candidates)
        w1, h1 = before[:, 2] - before[:, 0], before[:, 3] - before[:, 1]
        w2, h2 = after[:, 2] - after[:, 0], after[:, 3] - after[:, 1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        keep = (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > self.mosaic_area_thr) & (ar < 100)

        return {
            "im_file": first["im_file"],
            "img": img,
            "ori_shape": (th, tw),  # mosaic has no "original" image: identity mapping
            "resized_shape": (th, tw),
            "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
            "cls": cls[keep],
            "instances": instances[keep],
        }

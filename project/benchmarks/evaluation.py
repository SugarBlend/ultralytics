import gc
import json
import logging
import os
import sys
from multiprocessing import freeze_support
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv(f"{Path(__file__).parents[1]}/train/.env")
sys.path.insert(0, Path(__file__).parents[2].as_posix())

from project.benchmarks.rect_validator import RectValidator
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.checks import check_imgsz

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# getLogger, not logging.Logger(): a directly constructed Logger has no parent,
# ignores basicConfig handlers and silently drops INFO messages
logger = logging.getLogger(__name__)

Mode = str | tuple[str, int]  # "letterbox" | "mosaic" | ("collage", grid)


def _mb(x: int) -> float:
    return round(x / 2**20, 1)


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def mode_name(mode: str, grid: int = 2) -> str:
    return f"collage{grid}x{grid}" if mode == "collage" else mode


def validate(
    device: str,
    data: str,
    weights: str,
    hw: tuple[int, int],
    mode: str = "letterbox",
    grid: int = 2,
    batch: int = 8,
    max_det: int = 1000,
    **extra,
):
    h, w = hw
    overrides = dict(
        device=device, data=data, imgsz=max(hw), batch=batch, mosaic=1.0 if mode == "mosaic" else 0.0,
        rect=False, split="val", seed=0, plots=True,
        # a collage holds grid*grid source images in one frame (CrowdHuman: 300+ people each),
        # so the per-frame detection limit has to grow with it, otherwise recall is cut artificially
        max_det=max_det * (grid**2 if mode == "collage" else 1),
        name=f"{Path(weights).stem}_{h}x{w}_{mode_name(mode, grid)}",
        **extra,
    )
    args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)  # fresh args every run: the validator mutates them
    cuda = torch.cuda.is_available() and str(device) != "cpu"
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    validator = RectValidator(args=args)
    validator.target_hw = hw
    validator.val_mode = mode
    validator.grid = grid

    yolo = YOLO(weights)
    stats = validator(model=yolo.model)
    metrics = {k: float(v) for k, v in stats.items()}
    speed = {k: float(v) for k, v in validator.speed.items()}
    if mode == "collage":
        # a single collage frame contains a grid*grid array of source images: normalize to ms per source image
        # so that the speed comparison with letterbox/mosaic is fair
        speed = {k: v / grid**2 for k, v in speed.items()}

    memory = None
    if cuda:
        memory = {
            "peak_allocated_mb": _mb(torch.cuda.max_memory_allocated()),
            "peak_reserved_mb": _mb(torch.cuda.max_memory_reserved()),
        }

    # persistent dataloader workers otherwise live until the loader is garbage-collected
    dl = getattr(validator, "dataloader", None)
    if dl is not None and hasattr(dl, "close"):
        dl.close()

    del validator, yolo, stats, dl
    release_memory()

    if cuda:
        memory["after_cleanup_allocated_mb"] = _mb(torch.cuda.memory_allocated())
        memory["after_cleanup_reserved_mb"] = _mb(torch.cuda.memory_reserved())
        LOGGER.info(
            f"{colorstr('gpu mem:')} peak {memory['peak_allocated_mb']} MB allocated / "
            f"{memory['peak_reserved_mb']} MB reserved; after cleanup "
            f"{memory['after_cleanup_allocated_mb']} / {memory['after_cleanup_reserved_mb']} MB"
        )
        if memory["after_cleanup_allocated_mb"] > 50:
            LOGGER.warning(
                f"{memory['after_cleanup_allocated_mb']} MB still allocated after cleanup: "
                "something keeps references to GPU tensors between runs"
            )
    return metrics, speed, memory


def _run_key(model: str, hw, mode: str) -> tuple:
    return model, tuple(hw), mode


def process(
    device: str,
    data: str,
    models: dict[str, list[str]],
    resolutions: list[int | list[int]],
    modes: list[Mode],
    out_path: Path,
    batch: int = 8,
) -> None:
    # resume: keep what is already in out_path and skip finished runs, so a crash (e.g. MinIO timeout)
    # neither loses earlier results nor forces to recompute them
    results: dict[str, list[dict]] = {}
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
    done = {_run_key(r["model"], r["imgsz"], r["mode"]) for rows in results.values() for r in rows}
    failed = []

    for family, weights_list in models.items():
        for weights in weights_list:
            for imgsz in resolutions:
                hw = tuple(check_imgsz(imgsz, min_dim=2, max_dim=2))
                for m in modes:
                    mode, grid = (m, 2) if isinstance(m, str) else m
                    name = mode_name(mode, grid)
                    if _run_key(weights, hw, name) in done:
                        logger.info(f"skip (already in {out_path.name}): {weights} {hw[0]}x{hw[1]} {name}")
                        continue
                    try:
                        metrics, speed, memory = validate(device, data, weights, hw, mode, grid, batch)
                    except Exception as e:  # one broken run must not kill the whole benchmark
                        logger.exception(f"FAILED: {weights} {hw[0]}x{hw[1]} {name}: {e}")
                        failed.append((weights, hw, name))
                        release_memory()
                        continue
                    results.setdefault(family, []).append(
                        {
                            "model": weights,
                            "imgsz": list(hw),
                            "mode": name,
                            "metrics": metrics,
                            "bandwidth": speed,
                            "memory": memory,
                        }
                    )
                    done.add(_run_key(weights, hw, name))
                    # rewrite after every run so nothing is lost on a crash
                    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # short summary table
    logger.info(f"\n{'model':<16}{'HxW':>11}  {'mode':<12}{'mAP50':>8}{'mAP50-95':>10}")
    for rows in results.values():
        for r in rows:
            m = r["metrics"]
            logger.info(
                f"{Path(r['model']).stem:<16}{r['imgsz'][0]:>5}x{r['imgsz'][1]:<5}  {r['mode']:<12}"
                f"{m.get('metrics/mAP50(B)', float('nan')):>8.4f}{m.get('metrics/mAP50-95(B)', float('nan')):>10.4f}"
            )
    if failed:
        logger.warning(f"{len(failed)} run(s) failed, rerun the script to retry them: {failed}")


if __name__ == "__main__":
    freeze_support()

    models = {
        "yolov5": ['yolov5nu.pt', 'yolov5su.pt', 'yolov5mu.pt', 'yolov5lu.pt', 'yolov5xu.pt'],
        "yolov8": [
            'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
            r"C:\Users\Alexander\Downloads\pretrained_yolos\yolov8s\yolov8s_random_mosaic.pt",
            r"C:\Users\Alexander\Downloads\pretrained_yolos\yolov8l\yolov8l_random_mosaic.pt"
        ],
        "yolov9": ['yolov9t.pt', 'yolov9s.pt', 'yolov9m.pt', 'yolov9c.pt', 'yolov9e.pt'],
        "yolov10": ['yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt', 'yolov10l.pt', 'yolov10x.pt'],
        "yolo11": ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'],
        "yolo12": ['yolo12n.pt', 'yolo12s.pt', 'yolo12m.pt', 'yolo12l.pt', 'yolo12x.pt'],
        "yolo26": [
            'yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', 'yolo26l.pt', 'yolo26x.pt',
            r"C:\Users\Alexander\Downloads\pretrained_yolos\yolo26l\yolov26l_clean_mosaic.pt",
            r"C:\Users\Alexander\Downloads\pretrained_yolos\yolo26l\yolo26l_random_mosaic.pt"
        ],
    }

    process(
        device="0",
        data=r"E:\Projects\ultralytics\project\train\custom.yaml",
        models=models,
        resolutions=[[384, 640], [704, 1280]],
        modes=["letterbox", "mosaic", ("collage", 2)],
        out_path=Path("benchmark_COCO.json"),
        batch=8,
    )
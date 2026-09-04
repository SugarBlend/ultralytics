import argparse
import os
import json
import logging
from pathlib import Path
from label_studio_sdk import LabelStudio
import time
from typing import Any
from urllib.parse import unquote
from dotenv import load_dotenv
import re
from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", default="INFO")), format="[%(levelname)s] %(message)s")


class LSConverter(object):
    def __init__(self, url: str | None = None, api_key: str | None = None) -> None:
        self.client = LabelStudio(
            base_url=url or os.getenv("LABEL_STUDIO_URL"),
            api_key=api_key or os.getenv("LABEL_STUDIO_API_KEY")
        )

    def tasks2json(self, tasks, project_name, output_dir="outputs"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pb = tqdm(total=len(tasks), desc="Progress", leave=True, ncols=80, unit="task")
        for task in tasks:
            image_name = Path(unquote(task["data"]["image"])).relative_to(args.root_dataset_path).as_posix()

            if not task.get(args.data_type):
                logger.debug(f"No {args.data_type} for project: '{project_name}', task {pb.n}, '{task['data']['image']}'")
                continue

            for annotation in task[args.data_type]:
                if annotation.get("was_cancelled"):
                    logger.debug(f"Skipping cancelled annotation for task {pb.n}, '{image_name}'")
                    continue

                results = annotation["result"]
                boxes = {}

                # Sorting each bounding boxes and search them children by using parentID connection
                for label in results:
                    if label["type"] in ("rectanglelabels", "labels"):
                        boxes[label["id"]] = label

                detections = []
                for box_id, box_label in boxes.items():
                    category_name = None
                    key = box_label["type"]
                    if len(box_label["value"].get(key, [])) > 0:
                        category_name = box_label["value"][key][0]

                    if category_name is None:
                        logger.warning(f"Unknown label or empty for image {image_name}")
                        continue

                    width = box_label.get("original_width")
                    height = box_label.get("original_height")
                    if width is None or height is None:
                        logger.warning(f"Width / height missing for {image_name}")
                        continue

                    self.process_rectangle(box_label, detections, (width, height))

                output_file = output_dir.joinpath(project_name, Path(image_name).stem).with_suffix(".txt")
                output_file.parent.mkdir(exist_ok=True, parents=True)
                with output_file.open(mode="w") as file:
                    file.writelines(detections)

            pb.update()

    @staticmethod
    def process_rectangle(label: dict[str, Any], detections: list[str], frame_shape: tuple[int, int]) -> None:
        frame_w, frame_h = frame_shape
        value = label["value"]
        w = value["width"] * label["original_width"] / 100
        h = value["height"] * label["original_height"] / 100
        x0 = value["x"] * label["original_width"] / 100
        y0 = value["y"] * label["original_height"] / 100

        detections.append(' '.join(map(str, [0, (x0 + w / 2) / frame_w, (y0 + h / 2) / frame_h, w / frame_w, h / frame_h])))

    def _export_project_annotations(self, project_id: int) -> list[dict[str, Any]]:
        export_job = self.client.projects.exports.create(
            id=project_id,
            title=f"Export_{project_id}",
            serialization_options={
                "predictions": {"only_id": False},
                "drafts": {"only_id": False},
            },
        )
        export_id = export_job.id
        logger.info(f"Snapshot export created: {export_id}")

        while True:
            job = self.client.projects.exports.get(id=project_id, export_pk=export_id)
            if job.status == "completed":
                logger.info(f"Export completed: {export_id}")
                break
            elif job.status == "failed":
                raise RuntimeError(f"Export failed for project {project_id}")
            else:
                logger.info(f"Waiting for export {export_id}, status: {job.status}")
                time.sleep(2)

        tasks = self.client.projects.exports.download(id=project_id, export_pk=export_id, export_type="JSON")
        tasks_json = json.loads(b"".join(tasks).decode("utf-8"))
        return tasks_json

    def process_annotations(self, output_dir: str = "outputs") -> None:
        compilers = [re.compile(pattern) for pattern in args.patterns.split(',')]
        for project in self.client.projects.list().items:
            if any(compiler.search(project.title)for compiler in compilers):
                counter = project.total_annotations_number if args.data_type == "annotations" else project.total_predictions_number
                if not counter:
                    logger.warning(f"Skip empty project: '{project.title}', doesn't detect any {args.data_type}.")
                    continue

                logger.info(f"Processing project: {project.title} (id={project.id})")
                # Fetch label config directly from project settings
                tasks = self._export_project_annotations(project.id)
                self.tasks2json(tasks, project.title, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Label Studio projects to COCO via snapshot export")
    parser.add_argument("--root-dataset-path", default="s3://datasets",
                        help="The base path relative to which paths to the data described in the annotation are constructed.")
    parser.add_argument("--output_folder", default="labels",
                        help="Folder to save COCO JSONs")
    # parser.add_argument("--patterns", default=".*",
    #                     help="Regular expression for filtering project by them name.")
    parser.add_argument("--patterns", default="^Pose Annotation,^Foots Pose Annotation",
                        help="Regular expression for filtering project by them name.")
    # parser.add_argument("--patterns", default=r"Hands\s\+\sBody Pose Annotation",
    #                     help="Regular expression for filtering project by them name.")
    parser.add_argument("--data_type", choices=["predictions", "annotations"], default="annotations",
                        help="'Predictions' are the type of data obtained from LS from an auto-labeler, "
                             "'annotations' are data from LS that are marked up by people. ")
    args = parser.parse_args()

    load_dotenv()
    converter = LSConverter()
    converter.process_annotations(output_dir=args.output_folder)

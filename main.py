import os
import shutil
from pathlib import Path

from src.processing.merge_tracks import merge_tracks
from src.processing.utils import apply_func_to_dir

project_dir = Path(__file__).resolve().parents[0]
annotations_folder = os.path.join(project_dir, "data", "annotations")
raw_folder = os.path.join(project_dir, "data", "raw")

if os.path.exists(raw_folder):
    shutil.rmtree(raw_folder)

apply_func_to_dir(
    func=merge_tracks,
    project_dir=project_dir,
    input_folder="annotations",
    output_folder="raw",
    filter=lambda files: len(files) > 1,
)

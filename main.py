import os
import shutil
from pathlib import Path, PurePath

import yaml
from box import Box

from src.processing.transforms import Compose, ToDataFrame, ToFeatures, ToMuspy

with open("config.yaml", "r") as file:
    config = Box(yaml.safe_load(file))

cwd = Path.cwd()
annotations_dir = cwd.joinpath(config.paths.annotations)
raw_dir = cwd.joinpath(config.paths.raw)
muspy_dir = cwd.joinpath(config.paths.muspy)
dataframe_dir = cwd.joinpath(config.paths.dataframe)
features_dir = cwd.joinpath(config.paths.features)


if os.path.exists(raw_dir):
    shutil.rmtree(raw_dir)
if not os.path.exists(raw_dir):
    os.makedirs(muspy_dir)
    os.makedirs(dataframe_dir)
    os.makedirs(features_dir)

processing = Compose(
    [
        ToMuspy(to_file=True, output_path=muspy_dir),
        ToDataFrame(to_file=True, output_path=dataframe_dir),
        ToFeatures(
            features_processing=config.features_processing,
            to_file=True,
            output_path=features_dir,
        ),
    ]
)

for cur_path, directories, files in os.walk(annotations_dir):
    if len(files) > 0:
        input_path = PurePath(cur_path)
        song_name = f"{input_path.parent.parent.name} - {input_path.name}"
        processing(data=cur_path, filename=song_name)

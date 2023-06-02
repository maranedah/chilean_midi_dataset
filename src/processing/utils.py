import os
import shutil
from typing import Callable


def apply_func_to_dir(
    func: Callable,
    project_dir: str,
    input_folder: str,
    output_folder: str,
    filter: Callable,
    keep_original_folder_structure: bool = False,
):
    input_path = os.path.join(project_dir, "data", input_folder)
    for cur_path, directories, files in os.walk(input_path):
        if filter(files):
            input_path = cur_path
            if keep_original_folder_structure:

                def ignore_files(dir, files):
                    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

                shutil.copytree(
                    input_path,
                    input_path.replace(input_folder, output_folder),
                    ignore=ignore_files,
                )
                output_path = cur_path.replace(input_folder, output_folder)
            else:
                output_path = os.path.join(project_dir, "data", output_folder)
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
            func(input_path=input_path, output_path=output_path)

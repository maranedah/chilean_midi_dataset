import os
import shutil
from pathlib import Path

from src.utils.merge_tracks import merge_tracks

project_dir = Path(__file__).resolve().parents[0]
annotations_folder = os.path.join(project_dir, "data", "annotations")
raw_folder = os.path.join(project_dir, "data", "raw")

def ignore_files(dir, files):
	return [f for f in files if os.path.isfile(os.path.join(dir, f))]
 
shutil.rmtree(raw_folder)
shutil.copytree(annotations_folder, raw_folder, ignore=ignore_files)

for artist_folder in os.listdir(annotations_folder):
	albums_path = os.path.join(annotations_folder, artist_folder)
	for album_folder in os.listdir(albums_path):
		songs_path = os.path.join(albums_path, album_folder)
		for song_folder in os.listdir(songs_path):
			input_path = os.path.join(annotations_folder, artist_folder, album_folder, song_folder)
			output_path = os.path.join(raw_folder, artist_folder, album_folder, song_folder)
			merge_tracks(input_path=input_path, song_name=song_folder, output_path=output_path)


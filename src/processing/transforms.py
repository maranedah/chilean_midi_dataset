import json
import os
from pathlib import PurePath

import muspy

from .compute_positions import get_tempos, get_time_signatures


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class ToMuspy:
    def __init__(self, to_file, output_path):
        self.to_file = to_file
        self.output_path = PurePath(output_path)

    def __call__(self, path):
        input_path = PurePath(path)
        song_name = f"{input_path.parent.parent.name} - {input_path.name}"
        files = [
            input_path.joinpath(f)
            for f in os.listdir(input_path)
            if PurePath(f).suffix == ".mid"
        ]

        song = muspy.Music(
            metadata=muspy.Metadata(title=song_name),
            resolution=384,  # resolution=96,
            tracks=[],
        )
        if len(files) < 2:
            return song

        for f in files:
            track = muspy.read_midi(f).tracks[0]
            track.name = PurePath(f).stem
            if "drum" in track.name:
                track.is_drum = True
            song.tracks.append(track)

        metadata = json.load(open(input_path.joinpath("metadata.json")))
        song.time_signatures = get_time_signatures(metadata, song)
        song.tempos = get_tempos(metadata, song)
        if self.to_file:
            song.write(self.output_path.joinpath(song_name + ".mid"))

        return song

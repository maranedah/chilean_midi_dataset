import json
import os
from pathlib import PurePath

import muspy

from .compute_positions import get_tempos, get_time_signatures


def merge_tracks(input_path: str, output_path: str):
    path = PurePath(input_path)
    song_name = f"{path.parent.parent.name} - {path.name}"
    files = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.split(".")[-1] == "mid"
    ]
    if len(files) < 2:
        return
    first_instrument = muspy.read_midi(files[0])

    song = muspy.Music(
        metadata=muspy.Metadata(title=song_name),
        resolution=384,  # first_instrument.resolution,
        key_signatures=first_instrument.key_signatures,
        beats=first_instrument.beats,
        lyrics=None,
        annotations=None,
        tracks=[],
    )

    for f in files:
        track = muspy.read_midi(f).tracks[0]
        track.name = f.split(input_path)[-1].split(".")[0]
        if "drum" in f.lower():
            track.is_drum = True
        song.tracks.append(track)

    metadata = json.load(open(os.path.join(input_path, "metadata.json")))
    song.time_signatures = get_time_signatures(metadata, song)
    song.tempos = get_tempos(metadata, song)

    song.write(os.path.join(output_path, song_name + ".mid"))

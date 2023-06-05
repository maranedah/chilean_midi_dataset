import json
import os
from pathlib import PurePath

import muspy


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

    metadata = json.load(open(os.path.join(input_path, "metadata.json")))

    prev_end_time = 0
    time_signatures = []
    measures = []
    for i, ts in enumerate(metadata["time_signatures"]):
        time_signatures.append(
            muspy.TimeSignature(
                time=int(prev_end_time),
                numerator=int(ts["time_signature"].split("/")[0]),
                denominator=int(ts["time_signature"].split("/")[1]),
            )
        )
        if i + 1 == len(metadata["time_signatures"]):
            break
        ts_1 = metadata["time_signatures"][i + 1]
        for j in range(ts["measure"], ts_1["measure"]):
            measures.append(muspy.Barline(time=prev_end_time))
            prev_end_time += (first_instrument.resolution / 4) * (
                time_signatures[-1].numerator / (time_signatures[-1].denominator / 4)
            )
    breakpoint()

    tempos = [
        muspy.Tempo(
            time=(t["measure"] - 1) * first_instrument.resolution, qpm=int(t["tempo"])
        )
        for t in metadata["tempos"]
    ]

    song = muspy.Music(
        metadata=muspy.Metadata(title=song_name),
        resolution=first_instrument.resolution,
        tempos=tempos,
        key_signatures=first_instrument.key_signatures,
        time_signatures=time_signatures,
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
    song.write(os.path.join(output_path, song_name + ".mid"))

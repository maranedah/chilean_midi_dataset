import json
import os
from pathlib import PurePath

import muspy


def get_time_signatures(metadata, song):
    prev_end_time = 0
    time_signatures = []
    time_signature_list = metadata["time_signatures"]
    time_signature_list.append(
        {"time_signature": None, "measure": -1}
    )  # dummy last entry

    for ts, ts_next in zip(time_signature_list, time_signature_list[1:]):
        numerator, denominator = map(int, ts["time_signature"].split("/"))
        time_signatures.append(
            muspy.TimeSignature(
                time=int(prev_end_time),
                numerator=int(numerator),
                denominator=int(denominator),
            )
        )
        n_measures = ts_next["measure"] - ts["measure"]
        prev_end_time += song.resolution * (numerator / denominator) * n_measures

    return time_signatures


def get_tempos(metadata, song):
    pass


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

    song = muspy.Music(
        metadata=muspy.Metadata(title=song_name),
        resolution=first_instrument.resolution,
        # tempos=tempos,
        key_signatures=first_instrument.key_signatures,
        # time_signatures=time_signatures,
        # barlines=measures,
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

    # Get measures data

    """tempos = [
        muspy.Tempo(
            time=get_time_measure(t["measure"]),
            qpm=int(t["tempo"])

            #time=(t["measure"] - 1) * song.resolution, qpm=int(t["tempo"])
        )
        for t in metadata["tempos"]
    ]"""

    song.time_signatures = get_time_signatures(metadata, song)
    song.tempos = get_tempos(metadata, song)
    breakpoint()
    # song.tempos = tempos
    # song.barlines = measures

    song.write(os.path.join(output_path, song_name + ".mid"))

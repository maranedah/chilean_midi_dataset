import pandas as pd


def get_time_signature(song, time):
    ordered_ts = [
        f"{ts.numerator}/{ts.denominator}"
        for ts in song.time_signatures
        if ts.time <= time
    ]
    return ordered_ts[-1]


def get_tempo(song, time):
    ordered_tempo = [int(ts.qpm) for ts in song.tempos if ts.time <= time]
    return ordered_tempo[-1]


def get_measure_length(song, time):
    ordered_ts = [
        (song.resolution / 4) * (ts.numerator / (ts.denominator / 4))
        for ts in song.time_signatures
        if ts.time <= time
    ]
    return ordered_ts[-1]


def muspy_to_df(song):
    df = pd.DataFrame(
        [
            {
                "instrument": track.name,
                "track": i,
                # TODO: measure
                "time": note.time,
                # TODO: relative_position
                "measure_length": get_measure_length(
                    song, note.time
                ),  # TODO: measure_length
                "duration": note.duration,
                "pitch": note.pitch,
                "velocity": note.velocity,
                "tempo": get_tempo(song, note.time),  # TODO: tempo
                "time_signature": get_time_signature(
                    song, note.time
                ),  # TODO: time_signature
            }
            for i, track in enumerate(song.tracks)
            for note in track.notes
        ]
    )
    return df


if __name__ == "__main__":
    import muspy

    song = muspy.read_midi("../../data/raw/Super Especial - Pienso.mid")
    # song = song.adjust_resolution(16)
    df = muspy_to_df(song)
    breakpoint()

import muspy
import numpy as np


class Processor:
    def __init__(self, song):
        self.measures = self.get_measures(song)

    def get_measures(self, song):
        time_signatures = song.deepcopy().time_signatures
        last_ts = song.deepcopy().time_signatures[-1]
        last_ts.time = song.get_end_time()
        time_signatures.append(last_ts)

        measures = []
        for ts, ts_next in zip(time_signatures, time_signatures[1:]):
            measures_length = int(song.resolution * (ts.numerator / ts.denominator))
            measure_times = range(ts.time, ts_next.time, measures_length)
            measures.extend(measure_times)

        return np.array(measures)

    def time_to_measure(self, time):
        index = np.searchsorted(self.measures, time + 1, side="left")
        # if index > 0:
        index -= 1
        relative_position = time - self.measures[index]
        return index, relative_position

    def measure_to_time(self, measure):
        return self.measures[measure]


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
    tempos = [
        muspy.Tempo(
            time=Processor(song).measure_to_time(t["measure"] - 1), qpm=int(t["tempo"])
        )
        for t in metadata["tempos"]
    ]
    return tempos

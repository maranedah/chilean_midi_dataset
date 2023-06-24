import muspy
import numpy as np


class Measure:
    def __init__(self, index, time, length, time_signature):
        self.index = index
        self.time = time
        self.length = length
        self.time_signature = time_signature

    def __repr__(self):
        return f"Measures({self.index},{self.time},{self.length},{self.time_signature})"


class MuspyWithMeasures(muspy.Music):
    def __init__(self, song, metadata):
        super().__init__(**song.__dict__)
        self.extra_metadata = metadata
        self.time_signatures = self.get_time_signatures(metadata, song)
        self.measures = self.get_measures(song)
        self.tempos = self.get_tempos()

    def get_time_signatures(self, metadata, song):
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

    def get_measures(self, song):
        time_signatures = self.time_signatures
        last_ts = muspy.TimeSignature(
            numerator=4, denominator=4, time=song.get_end_time()
        )  # dummy last time signature
        time_signatures.append(last_ts)

        measures = []
        j = 0
        for ts, ts_next in zip(time_signatures, time_signatures[1:]):
            measures_length = int(song.resolution * (ts.numerator / ts.denominator))
            measure_times = range(ts.time, ts_next.time, measures_length)
            measures = [
                Measure(
                    index=i + j, time=time, length=measures_length, time_signature=ts
                )
                for i, time in enumerate(measure_times)
            ]
            j += len(measures)
            measures.extend(measures)

        return measures

    def get_tempos(self):
        tempos = [
            muspy.Tempo(
                time=self.measure_to_time(t["measure"] - 1), qpm=int(t["tempo"])
            )
            for t in self.extra_metadata["tempos"]
        ]
        return tempos

    def time_to_measure(self, time):
        index = np.searchsorted(self.measures, time + 1, side="left") - 1
        relative_position = time - self.measures[index]
        return index, relative_position

    def measure_to_time(self, measure):
        return self.measures[measure].time

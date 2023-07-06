import json
import os
from pathlib import PurePath

import muspy
import numpy as np
import pandas as pd

from .encoders import IdentityEncoder, LabelEncoder, RangeEncoder
from .quantizer import MuspyWithMeasures


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data, filename):
        for t in self.transforms:
            data = t(data, filename)
        return data


class ToMuspy:
    def __init__(self, to_file, output_path):
        self.to_file = to_file
        self.output_path = PurePath(output_path)

    def __call__(self, path, song_name):
        input_path = PurePath(path)
        files = [
            input_path.joinpath(f)
            for f in os.listdir(input_path)
            if PurePath(f).suffix == ".mid"
        ]

        song = muspy.Music(
            metadata=muspy.Metadata(title=song_name),
            resolution=96,  # resolution=96,
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

        # quantize
        metadata = json.load(open(input_path.joinpath("metadata.json")))
        song = MuspyWithMeasures(song, metadata)

        if self.to_file:
            song.write(self.output_path.joinpath(song_name + ".mid"))

        return song


class ToDataFrame:
    def __init__(self, to_file, output_path):
        self.to_file = to_file
        self.output_path = PurePath(output_path)

    def __call__(self, data, song_name):
        df = pd.DataFrame(
            [
                {
                    "instrument": track.name,
                    "track": i,
                    "measure": data.time_to_measure(note.time).index,
                    "time": note.time,
                    "relative_position": note.time
                    - data.time_to_measure(note.time).time,
                    "measure_length": data.time_to_measure(note.time).length,
                    "duration": note.duration,
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                    "tempo": int(data.time_to_measure(note.time).tempo.qpm),
                    "time_signature": data.time_signature_str(
                        data.time_to_measure(note.time).time_signature
                    ),
                }
                for i, track in enumerate(data.tracks)
                for note in track.notes
            ]
        )
        df = df.sort_values(by=["time", "instrument"])
        if self.to_file:
            df.to_csv(self.output_path.joinpath(song_name + ".csv"), index=False)
        return df


class ToFeatures:
    def __init__(self, features_processing, to_file, output_path):
        self.features_processing = features_processing
        self.to_file = to_file
        self.output_path = PurePath(output_path)
        self.encoders_dict = {
            "LabelEncoder": LabelEncoder,
            "IdentityEncoder": IdentityEncoder,
            "RangeEncoder": RangeEncoder,
        }

    def __call__(self, data, song_name):
        df = data[self.features_processing.keys()]

        data = None
        if os.path.exists("encoders.json"):
            with open("encoders.json", "r") as f:
                data = json.load(f)

        labels = []
        encoders = {}
        for feature_name in self.features_processing.keys():
            feature = self.features_processing[feature_name]
            encoder = self.encoders_dict[feature.encoder.name](
                label_mapping=data[feature_name]["label_mapping"] if data else {},
                inverse_mapping=data[feature_name]["inverse_mapping"] if data else {},
                **feature.encoder.params,
            )
            feature_labels = encoder.fit_transform(df[feature_name])
            labels.append(feature_labels)
            encoders[feature_name] = {
                "label_mapping": encoder.label_mapping,
                "inverse_mapping": encoder.inverse_mapping,
            }

        with open("encoders.json", "w") as f:
            json.dump(encoders, f)

        features = np.vstack(labels).T
        np.save(
            file=self.output_path.joinpath(song_name + ".npy"),
            arr=features,
            allow_pickle=True,
        )
        return features


class BackToDataFrame:
    def __init__(self, features_processing, to_file, output_path):
        self.features_processing = features_processing
        self.encoders_dict = {
            "LabelEncoder": LabelEncoder,
            "IdentityEncoder": IdentityEncoder,
            "RangeEncoder": RangeEncoder,
        }

    def __call__(self, data, song_name):
        with open("encoders.json", "r") as f:
            self.encoders_data = json.load(f)

        df = pd.DataFrame(data, columns=self.features_processing.keys())
        for feature_name in self.features_processing.keys():
            feature = self.features_processing[feature_name]
            encoder = self.encoders_dict[feature.encoder.name](
                label_mapping=self.encoders_data[feature_name]["label_mapping"]
                if len(data)
                else {},
                inverse_mapping=self.encoders_data[feature_name]["inverse_mapping"]
                if len(data)
                else {},
                **feature.encoder.params,
            )
            df[feature_name] = encoder.inverse_transform(df[feature_name])

            print(df)

        df["velocity"] = 90
        df["tempo"] = 120
        df_measure = df[["measure", "measure_length"]].drop_duplicates()
        df_measure["measure_position"] = (
            df_measure["measure_length"].cumsum() - df["measure_length"][0].item()
        )
        df = pd.merge(df, df_measure, on="measure")
        df["abs_position"] = df["measure_position"] + df["relative_position"]
        df["abs_position"] = df["abs_position"].astype(int)
        return df


class BackToMidi:
    def __init__(self, to_file, output_path):
        self.to_file = to_file
        self.output_path = output_path

    def __call__(self, df, song_name):
        song = muspy.Music(resolution=96)

        time_signatures = df[df["time_signature"] != df["time_signature"].shift()]
        song.time_signatures = [
            muspy.TimeSignature(
                time=row["abs_position"],
                numerator=int(row["time_signature"].split("/")[0]),
                denominator=int(row["time_signature"].split("/")[1]),
            )
            for i, row in time_signatures.iterrows()
        ]

        tempos = df[df["tempo"] != df["tempo"].shift()]
        song.tempos = [
            muspy.Tempo(time=row["abs_position"], qpm=row["tempo"])
            for i, row in tempos.iterrows()
        ]

        song.tracks = []
        instruments = df["instrument"].drop_duplicates().to_numpy()
        for i, instrument in enumerate(instruments):
            track = muspy.Track(program=i, is_drum="drums" in instrument)
            notes = [
                muspy.Note(
                    time=note["abs_position"],
                    pitch=note["pitch"],
                    duration=note["duration"],
                    velocity=note["velocity"],
                )
                for _, note in df[df["instrument"] == instrument].iterrows()
            ]
            track.notes = notes

            song.tracks.append(track)

        song.write("file.mid")

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

    def __call__(self, data):
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
        if self.to_file:
            df.to_csv(
                self.output_path.joinpath(data.metadata.title + ".csv"), index=False
            )
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

    def __call__(self, data):
        df = data[self.features_processing]

        data = None
        if os.path.exists("encoders.json"):
            with open("encoders.json", "r") as f:
                data = json.load(f)

        labels = []
        encoders = []
        for feature_name in self.features_processing:
            feature = self.features_processing[feature_name]
            encoder = self.encoders_dict[feature.encoder.name](
                label_mapping=data[feature_name]["label_mapping"] if data else {},
                inverse_mapping=data[feature_name]["inverse_mapping"] if data else {},
                **feature.encoder.params,
            )
            feature_labels = encoder.fit_transform(df[feature_name])
            labels.append(feature_labels)
            encoders.append(
                {
                    feature_name: {
                        "label_mapping": encoder.label_mapping,
                        "inverse_mapping": encoder.inverse_mapping,
                    }
                }
            )

        with open("encoders.json", "w") as f:
            json.dump(encoders, f)

        features = np.vstack(labels).T
        return features

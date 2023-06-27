# made a new class so that calling fit doesn't reset the internal state of the encoder
class LabelEncoder:
    def __init__(self, label_mapping={}, inverse_mapping={}):
        self.label_mapping = label_mapping
        self.inverse_mapping = inverse_mapping

    def fit(self, labels):
        unique_labels = list(set(labels) - set(self.label_mapping.keys()))
        prev_labels = list(self.label_mapping.keys())

        for idx, label in enumerate(unique_labels):
            self.label_mapping[label] = len(prev_labels) + idx
            self.inverse_mapping[idx] = label
        return self

    def transform(self, labels):
        encoded_labels = []
        for label in labels:
            encoded_labels.append(self.label_mapping[label])
        return encoded_labels

    def inverse_transform(self, encoded_labels):
        original_labels = []
        for encoded_label in encoded_labels:
            original_labels.append(self.inverse_mapping[encoded_label])
        return original_labels

    def fit_transform(self, labels):
        self.fit(labels)
        return self.transform(labels)


class IdentityEncoder:
    def __init__(self, label_mapping={}, inverse_mapping={}):
        self.label_mapping = label_mapping
        self.inverse_mapping = inverse_mapping

    def fit(self, labels):
        return self

    def transform(self, labels):
        return labels

    def inverse_transform(self, encoded_labels):
        return encoded_labels

    def fit_transform(self, labels):
        return labels


class RangeEncoder:
    def __init__(self, label_mapping={}, inverse_mapping={}, range=1):
        self.label_mapping = label_mapping
        self.inverse_mapping = inverse_mapping
        self.range = range

    def fit(self, labels):
        return self

    def transform(self, labels):
        return labels // self.range

    def inverse_transform(self, encoded_labels):
        return encoded_labels * self.range

    def fit_transform(self, labels):
        return labels // self.range

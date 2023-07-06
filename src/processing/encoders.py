# made a new class so that calling fit doesn't reset the internal state of the encoder
class LabelEncoder:
    def __init__(self, label_mapping={}, inverse_mapping={}):
        self.label_mapping = label_mapping
        self.inverse_mapping = inverse_mapping

    def fit(self, labels):
        unique_labels = list(set(labels) - set(self.label_mapping.keys()))
        prev_labels = [x for x in self.label_mapping.keys()]
        i = 0
        for label in unique_labels:
            if str(label) not in prev_labels:
                self.label_mapping[str(label)] = len(prev_labels) + i
                self.inverse_mapping[len(prev_labels) + i] = label
                i+=1
        print(self.label_mapping)
        
        return self

    def transform(self, labels):
        encoded_labels = []
        for label in labels:
            encoded_labels.append(self.label_mapping[str(label)])
        return encoded_labels

    def inverse_transform(self, encoded_labels):
        
        original_labels = []
        for encoded_label in encoded_labels:
            original_labels.append(self.inverse_mapping[str(encoded_label)])
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

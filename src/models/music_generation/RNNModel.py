import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.ModuleList([nn.Linear(hidden_size, n) for n in num_classes])

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        output = [fc(output) for fc in self.fc]
        return output, hidden

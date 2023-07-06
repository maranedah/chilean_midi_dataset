import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

import json

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.ModuleList([nn.Linear(hidden_size, n) for n in num_classes])

    def forward(self, x, hidden):
        #padded_sequences = pad_sequence(x, batch_first=True)
        #print(padded_sequences)
        #packed_sequences = pack_padded_sequence(padded_sequences, lengths=[len(seq) for seq in x], batch_first=True)
        #print(packed_sequences)
        output, hidden = self.rnn(x, hidden)
        #print(output)
        #output, _ = pad_packed_sequence(output, batch_first=True)
        output = [fc(output) for fc in self.fc]
        return output, hidden

    def fit(self, dataset):
        encoder = json.load(open("encoders.json"))
        output_size = [
            len(encoder[feature_labels]["label_mapping"].keys()) 
            for feature_labels in encoder.keys()
        ]

        #pack
        self(dataset, hidden=None)
        

        input_size = len(output_size)
        hidden_size = 512
        # Set the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)

        # Train the model
        num_epochs = 100
        batch_size = 1

        for epoch in range(num_epochs):
            hidden = None
            for i in range(0, dataset.shape[1] - batch_size - 1, batch_size):
                inputs = dataset[:, i + batch_size]
                targets = dataset[:, i + batch_size + 1] # target = next token
                optimizer.zero_grad()

                #breakpoint()
                outputs, hidden = self(inputs, hidden)

                losses = []
                for j, output in enumerate(outputs):
                
                    loss = criterion(
                        output.view(-1, output_size[j]), targets[:, j].view(-1).long()
                    )
                    losses.append(loss)
                loss = sum(losses)

                loss.backward()
                optimizer.step()

                hidden = hidden.detach()

                # Print the loss for monitoring
                if i % 100 == 0:
                    print(targets.int().tolist()[0])
                    print(
                        [
                            output.view(-1, output_size[j]).argmax(-1).item()
                            for j, output in enumerate(outputs)
                        ]
                    )
                    print(f"Epoch [{epoch+1}], Step [{i+1}], Loss: {loss.item():.4f}")
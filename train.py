import json

import numpy as np
import torch
import torch.nn as nn

from src.models.music_generation import RNNModel

# Set the dimensions

data = json.load(open("encoders.json"))


output_size = [
    len(data[feature_labels]["label_mapping"].keys()) for feature_labels in data.keys()
]
input_size = len(output_size)
hidden_size = 512

# breakpoint()
# output_size = [3,3,3,3,3,3,3,3,3]

# Create an instance of the model
model = RNNModel(input_size, hidden_size, output_size)
model = model.cuda("cuda:0")

# Convert the data to PyTorch tensors
input_sequences = (
    torch.tensor(
        np.load("data/raw/npy/Super Especial - 03 - Evasiva.npy"), dtype=torch.float32
    )
    .unsqueeze(0)
    .cuda()
)
# target_sequences = torch.tensor(target_sequences, dtype=torch.float32)

# Set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# Train the model
num_epochs = 100
batch_size = 1

for epoch in range(num_epochs):
    hidden = None
    for i in range(0, input_sequences.shape[1] - batch_size - 1, batch_size):
        inputs = input_sequences[:, i + batch_size]
        targets = input_sequences[:, i + batch_size + 1]
        # print(inputs)
        # print(targets)
        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)

        # loss = criterion(outputs.view(-1, output_size), targets.view(1, -1))
        losses = []
        for j, output in enumerate(outputs):
            # breakpoint()
            loss = criterion(
                output.view(-1, output_size[j]), targets[:, j].view(-1).long()
            )
            losses.append(loss)
        loss = sum(losses)
        # print(outputs.view(-1, output_size))

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

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


# Load the JSON files
metadata_path = 'filtered_chess_games.json'  # File with white_rating, black_rating, etc.
encoded_moves_path = 'encoded_chess_moves.json'  # File with encoded move sequences

metadata = pd.read_json(metadata_path)
encoded_moves = pd.read_json(encoded_moves_path, typ='series')

# Combine the data assuming the order is the same and the length of games is equal in both files
metadata = metadata.head(100)
metadata['encoded_moves'] = encoded_moves.values

# Input (X): Encoded move sequences
X = list(metadata['encoded_moves'])

# Output (Y): White and Black ratings
Y = metadata[['white_rating', 'black_rating']]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Find the maximum length of the sequences
max_sequence_len = max(len(game) for game in X)

X_train_flat = [[move for pair in game for move in pair] for game in X_train]
X_test_flat = [[move for pair in game for move in pair] for game in X_test]

def pad_sequences(sequences, maxlen, value=0):
    # Determine the number of features per step; in this case, it's 1 since each move is a single integer
    num_features = 1  # This is based on your encoded moves

    # Initialize the padded sequences with zeros or the specified value
    # The array should have a shape of (number of sequences, maxlen, number of features per step)
    padded_sequences = np.full((len(sequences), maxlen, num_features), value)

    # Copy the sequences into the padded array
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        for j in range(length):
            padded_sequences[i, j, 0] = seq[j]  # Assuming each `seq` is a flat list of integers

    return padded_sequences

# Pad the training and testing sequences
X_train_padded = pad_sequences(X_train_flat, maxlen=max_sequence_len)
X_test_padded = pad_sequences(X_test_flat, maxlen=max_sequence_len)


# Define the LSTM model
class ChessRatingPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ChessRatingPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)  # For white rating
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # For black rating

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        white_rating = self.fc1(lstm_out[:, -1, :])
        black_rating = self.fc2(lstm_out[:, -1, :])
        return white_rating, black_rating

# Set the input dimension to 1 since we have encoded moves as single features,
# and the output dimension to 1 for each rating prediction
input_dim = 1
hidden_dim = 64
output_dim = 1

model = ChessRatingPredictor(input_dim, hidden_dim, output_dim)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_padded).float()
Y_train_tensor = torch.tensor(Y_train.values).float()
X_test_tensor = torch.tensor(X_test_padded).float()
Y_test_tensor = torch.tensor(Y_test.values).float()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    white_output, black_output = model(X_train_tensor)
    loss_white = criterion(white_output.squeeze(), Y_train_tensor[:, 0])
    loss_black = criterion(black_output.squeeze(), Y_train_tensor[:, 1])
    loss = loss_white + loss_black
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    white_output, black_output = model(X_test_tensor)
    test_loss_white = criterion(white_output.squeeze(), Y_test_tensor[:, 0])
    test_loss_black = criterion(black_output.squeeze(), Y_test_tensor[:, 1])
    print(f'Test Loss White: {test_loss_white.item()}, Test Loss Black: {test_loss_black.item()}')




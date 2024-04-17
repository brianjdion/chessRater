import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from chess_model import ChessRatingPredictor  
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import joblib


# Load the JSON files
chess_games_path = 'filtered_chess_games.json'  # File with white_rating, black_rating, etc.
encoded_moves_path = 'encoded_chess_moves.json'  # File with encoded move sequences

chess_games = pd.read_json(chess_games_path)
encoded_moves = pd.read_json(encoded_moves_path, typ='series')

# Combine the data assuming the order is the same and the length of games is equal in both files
chess_games = chess_games.head(10000)
chess_games['encoded_moves'] = encoded_moves.values

ratings = chess_games[['white_rating', 'black_rating']]
moves = list(chess_games['encoded_moves'])

# Normalize ratings
scaler = StandardScaler()
ratings_scaled = scaler.fit_transform(ratings)

# Split the data into training and testing sets for both ratings and moves
X_train_moves, X_test_moves, Y_train, Y_test = train_test_split(moves, ratings_scaled, test_size=0.2, random_state=42)

# Flatten each sequence of moves
X_train_moves_flat = [move for seq in X_train_moves for move in seq]
X_test_moves_flat = [move for seq in X_test_moves for move in seq]

# Find the maximum length of the sequences
max_sequence_len = max(len(seq) for seq in X_train_moves)

def pad_sequences(sequences, maxlen, value=0):

    padded_sequences = np.full((len(sequences), maxlen), value)

    for i, game in enumerate(sequences):
        flat_game = [move for pair in game for move in pair]  # Flatten the move pairs
        length = min(len(flat_game), maxlen)
        padded_sequences[i, :length] = flat_game[:length]
    return padded_sequences

# Pad the training and testing move sequences
X_train_moves_padded = pad_sequences(X_train_moves, maxlen=max_sequence_len)
X_test_moves_padded = pad_sequences(X_test_moves, maxlen=max_sequence_len)

# Convert move sequences into tensor
X_train_moves_tensor = torch.tensor(X_train_moves_padded).float().unsqueeze(-1)  # Add feature dimension
X_test_moves_tensor = torch.tensor(X_test_moves_padded).float().unsqueeze(-1)  # Add feature dimension

# Convert ratings into tensors
Y_train_tensor = torch.tensor(Y_train).float()
Y_test_tensor = torch.tensor(Y_test).float()

# Prepare DataLoader for batch processing
batch_size = 64
train_dataset = TensorDataset(X_train_moves_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_moves_tensor, Y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = ChessRatingPredictor(move_input_dim=1, hidden_dim=64, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for moves, ratings in train_loader:
        optimizer.zero_grad()
        white_output, black_output = model(moves)
        loss_white = criterion(white_output, ratings[:, 0])
        loss_black = criterion(black_output, ratings[:, 1])
        loss = loss_white + loss_black
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss / len(train_loader)}')

# Evaluation
model.eval()
total_loss = 0
with torch.no_grad():
    for moves, ratings in test_loader:
        white_output, black_output = model(moves)
        loss_white = criterion(white_output, ratings[:, 0])
        loss_black = criterion(black_output, ratings[:, 1])
        loss = loss_white + loss_black
        total_loss += loss.item()
print(f'Test Loss: {total_loss / len(test_loader)}')


with torch.no_grad():
    white_output, black_output = model(X_test_moves_tensor)
    test_loss_white = criterion(white_output.squeeze(), Y_test_tensor[:, 0])
    test_loss_black = criterion(black_output.squeeze(), Y_test_tensor[:, 1])
    
    # Calculate the absolute differences
    white_loss = torch.abs(white_output.squeeze() - Y_test_tensor[:, 0])
    black_loss = torch.abs(black_output.squeeze() - Y_test_tensor[:, 1])

    # Elo range for accuracy calculation
    elo_range = 100

    # Count predictions within the Elo range
    within_range_white = torch.sum(white_loss <= elo_range).item()
    within_range_black = torch.sum(black_loss <= elo_range).item()

    # Calculate the accuracy
    total_predictions = Y_test_tensor.size(0)
    accuracy_white = within_range_white / total_predictions
    accuracy_black = within_range_black / total_predictions

    print(f'Test Loss White: {test_loss_white.item()}, Test Loss Black: {test_loss_black.item()}')
    print(f'Accuracy for white players within ±{elo_range} Elo: {accuracy_white}')
    print(f'Accuracy for black players within ±{elo_range} Elo: {accuracy_black}')

# Save the trained model
torch.save(model.state_dict(), 'chess_rating_predictor.pth')

# Save scaler
joblib.dump(scaler, 'elo_scaler.pkl')

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load the JSON files
chess_games_path = 'filtered_chess_games.json'  # File with white_rating, black_rating, etc.
encoded_moves_path = 'encoded_chess_moves.json'  # File with encoded move sequences

chess_games = pd.read_json(chess_games_path)
encoded_moves = pd.read_json(encoded_moves_path, typ='series')

# Combine the data assuming the order is the same and the length of games is equal in both files
chess_games = chess_games.head(100)
chess_games['encoded_moves'] = encoded_moves.values

ratings = chess_games[['white_rating', 'black_rating']]
moves = list(chess_games['encoded_moves'])

Y = chess_games[['white_rating', 'black_rating']]

# Split the data into training and testing sets for both ratings and moves
X_train_moves, X_test_moves, Y_train, Y_test = train_test_split(moves, Y, test_size=0.2, random_state=42)

# Flatten each sequence of moves
X_train_moves_flat = [move for seq in X_train_moves for move in seq]
X_test_moves_flat = [move for seq in X_test_moves for move in seq]

# Find the maximum length of the sequences
max_sequence_len = max(len(seq) for seq in X_train_moves)

def pad_sequences(sequences, maxlen, value=0):
    # Initialize the padded sequences
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

# Convert ratings into tensor
# X_train_ratings_tensor = torch.tensor(X_train_ratings.values).float()
# X_test_ratings_tensor = torch.tensor(X_test_ratings.values).float()

# Convert Y (target ratings) into tensors
Y_train_tensor = torch.tensor(Y_train.values).float()
Y_test_tensor = torch.tensor(Y_test.values).float()

class ChessRatingPredictor(nn.Module):
    def __init__(self, move_input_dim, rating_input_dim, hidden_dim, output_dim):
        super(ChessRatingPredictor, self).__init__()
        # LSTM for move sequences
        self.lstm = nn.LSTM(move_input_dim, hidden_dim, batch_first=True)
        
        # Fully connected layer for ratings
        self.fc_ratings = nn.Linear(rating_input_dim, hidden_dim)
        
        # Final fully connected layers for output
        self.fc1 = nn.Linear(hidden_dim * 2, output_dim)  # For white rating
        self.fc2 = nn.Linear(hidden_dim * 2, output_dim)  # For black rating

    def forward(self, moves):#, ratings):
        # LSTM for moves
        lstm_out, _ = self.lstm(moves)
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layer for ratings
        # ratings_out = self.fc_ratings(ratings)
        
        # Combine outputs from both paths
        # combined = torch.cat((lstm_out, ratings_out), dim=1)
        
        # Final output
        # white_rating = self.fc1(combined)
        # black_rating = self.fc2(combined)
        white_rating = self.fc1(lstm_out)
        black_rating = self.fc2(lstm_out)
        return white_rating, black_rating
    
move_input_dim = 1  # Each move is one-dimensional after encoding
rating_input_dim = 2  # Two ratings: white and black
hidden_dim = 64
output_dim = 1  # Predicting one rating at a time, but the model will have two outputs

model = ChessRatingPredictor(move_input_dim, rating_input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning rate can be adjusted

epochs = 50

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass: Compute predicted y by passing x to the model
    white_output, black_output = model(X_train_moves_tensor)#, X_train_ratings_tensor)
    
    # Compute and print loss
    loss_white = criterion(white_output.squeeze(), Y_train_tensor[:, 0])
    loss_black = criterion(black_output.squeeze(), Y_train_tensor[:, 1])
    total_loss = loss_white + loss_black
    
    # Zero gradients, perform a backward pass, and update the weights.
    total_loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}/{epochs}, Total Loss: {total_loss.item()}')

model.eval()

with torch.no_grad():
    white_output, black_output = model(X_test_moves_tensor)#, X_test_ratings_tensor)
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
# torch.save(model.state_dict(), 'chess_rating_predictor.pth')
def save(model, outname):
    torch.save(model.state_dict(), outname)

modelName = 'chessRater.pth'
save(model, modelName)



#### TESTING 

# move_sequence = [
#     [3501, 3505], [3445, 3447], [3503, 3395], [3389, 651], [790, 1268], [707, 957], 
#     [867, 873], [2304, 3426], [3426, 27], [62, 217], [1718, 1103], [1103, 1165], 
#     [3618, 719], [870, 1321], [65, 3564], [867, 76], [3677, 2799], [995, 3624], 
#     [2899, 796], [237, 3480], [2399, 52], [3167, 3376], [3621, 3535], [1124, 386], 
#     [1425, 1214], [337, 2536], [1767, 1649], [323, 3182], [809, 379], [1309, 2536]
# ]










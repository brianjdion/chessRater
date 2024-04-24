import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from chess_model import ChessELOPredictor, ChessNextMove
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from encode_moves import load_mappings

# Load the JSON files
chess_games_path = 'filtered_chess_games.json'  # File with white_rating, black_rating, etc.
encoded_moves_path = 'encoded_chess_moves.json'  # File with encoded move sequences

chess_games = pd.read_json(chess_games_path)
encoded_moves = pd.read_json(encoded_moves_path, typ='series')

# Combine the data assuming the order is the same and the length of games is equal in both files
chess_games = chess_games.head(10000)
chess_games['encoded_moves'] = encoded_moves.values

def get_ratings(chess_games):
    ratings = chess_games[['white_rating', 'black_rating']].values
    mean = np.mean(ratings.flatten())
    sd = np.std(ratings.flatten())
    ratings = (ratings - mean)/sd
    return ratings

# Let's flatten moves and make into input/output 
def flatten_pad_center(chess_games):
    
    games = chess_games['encoded_moves'].tolist()
    flat = []
    maxlen = 0
    for game in games:
        f = []
        for pair in game:
            f+=pair
        if len(f) > maxlen:
            maxlen = len(f)
        flat.append(f)

    padded = np.zeros((len(flat), maxlen))
    for idx, game in enumerate(flat):
        diff = maxlen - len(game)
        padded[idx,diff:] = game

    # get centered ELO
    ratings = torch.tensor(get_ratings(chess_games))
    dataset = TensorDataset(torch.tensor(padded).long(), ratings)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train, test

# Prepare DataLoader for batch processing
train, test = flatten_pad_center(chess_games)
batchSize = 64
train_loader = DataLoader(train, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(test, batch_size=batchSize, shuffle=False)


# Initialize the model, loss function, and optimizer
mappings = load_mappings()
model = ChessNextMove(max_moves = len(mappings)+1, hidden_dim= 600)
#model = ChessRatingPredictor(move_input_dim=1, hidden_dim=128, output_dim=1)
cost = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(1, epochs+1):
    total_loss = 0
    for batch_idx, (X, Y) in enumerate(train_loader):
        print(batch_idx + 1)
        if (batch_idx+1) % 100 == 0:
            break
        optimizer.zero_grad()
        hidden = model.init_hidden(batchSize)

        logits = model.forward(X, hidden)
#         Assuming logits are output at every sequence step and need reshaping:
        logits = logits.view(-1, logits.size(-1))  # Flatten output to fit [batch_size * seq_len, num_classes]
        loss = cost(logits, Y.view(-1))


        # loss = cost(logits.reshape(-1, len(mappings)+1), Y.flatten())
        total_loss += loss.item()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
    print(f'Epoch {epoch}/{epochs}, Total Loss: {total_loss / batch_idx}')

# Evalution
model.eval()
total_loss = 0
total_correct_white = 0
total_correct_black = 0
elo_range = 100  # Set the range for considering accurate predictions

for X, Y in test_loader:
    hidden = model.init_hidden(X.size(0))  # Adjust batch size dynamically
    with torch.no_grad():
        outputs = model(X, hidden)
        loss = cost(outputs, Y.float())
        total_loss += loss.item()

        # Calculate accuracy within ELO range
        predicted_white_elo, predicted_black_elo = outputs[:, 0], outputs[:, 1]
        true_white_elo, true_black_elo = Y[:, 0], Y[:, 1]

        # Absolute difference
        white_diff = torch.abs(predicted_white_elo - true_white_elo)
        black_diff = torch.abs(predicted_black_elo - true_black_elo)

        # Count correct predictions within the specified ELO range
        total_correct_white += torch.sum(white_diff <= elo_range).item()
        total_correct_black += torch.sum(black_diff <= elo_range).item()

total_games = len(test_loader.dataset)
accuracy_white = total_correct_white / total_games
accuracy_black = total_correct_black / total_games

print(f'Test Loss: {total_loss / len(test_loader)}')
print(f'Accuracy for white players within ±{elo_range} Elo: {accuracy_white}')
print(f'Accuracy for black players within ±{elo_range} Elo: {accuracy_black}')



###############################


# model = ChessRatingPredictor(move_input_dim=1, hidden_dim=128, output_dim=1)
# cost = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# epoch = 10
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for moves, ratings in train_loader:
#         optimizer.zero_grad()
#         white_output, black_output = model(moves)

#         loss_white = cost(white_output, ratings[:, 0])
#         loss_black = cost(black_output, ratings[:, 1])
#         loss = loss_white + loss_black
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f'Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss / len(train_loader)}')

# # Evaluation
# model.eval()
# total_loss = 0
# with torch.no_grad():
#     for moves, ratings in test_loader:
#         white_output, black_output = model(moves)
#         loss_white = cost(white_output, ratings[:, 0])
#         loss_black = cost(black_output, ratings[:, 1])
#         loss = loss_white + loss_black
#         total_loss += loss.item()
# print(f'Test Loss: {total_loss / len(test_loader)}')

# with torch.no_grad():
#     white_output, black_output = model(X_test_moves_tensor)
#     test_loss_white = cost(white_output.squeeze(), Y_test_tensor[:, 0])
#     test_loss_black = cost(black_output.squeeze(), Y_test_tensor[:, 1])
    
#     # Calculate the absolute differences
#     white_loss = torch.abs(white_output.squeeze() - Y_test_tensor[:, 0])
#     black_loss = torch.abs(black_output.squeeze() - Y_test_tensor[:, 1])

#     # Elo range for accuracy calculation
#     elo_range = 250

#     # Count predictions within the Elo range
#     within_range_white = torch.sum(white_loss <= elo_range).item()
#     within_range_black = torch.sum(black_loss <= elo_range).item()

#     # Calculate the accuracy
#     total_predictions = Y_test_tensor.size(0)
#     accuracy_white = within_range_white / total_predictions
#     accuracy_black = within_range_black / total_predictions

#     print(f'Test Loss White: {test_loss_white.item()}, Test Loss Black: {test_loss_black.item()}')
#     print(f'Accuracy for white players within ±{elo_range} Elo: {accuracy_white}')
#     print(f'Accuracy for black players within ±{elo_range} Elo: {accuracy_black}')


# Save the trained model
torch.save(model.state_dict(), 'chess_rating_predictor.pth')


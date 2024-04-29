import torch
from chess_model import ChessELOPredictor, ChessNextMove
import pandas as pd
import numpy as np
from encode_moves import load_mappings


# Load the trained model
mappings = load_mappings()
model = ChessELOPredictor('nextmove.pt', len(mappings)+1, 600)
model.load_state_dict(torch.load('chess_rating_predictor.pt'))

# Move sequence where each pair represents a turn
# This is a game between a 1547-1399
move_sequence_1 = [
    [1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
    [11, 12], [13, 14], [15, 16], [16, 17], [18, 19],
    [20, 21], [21, 22], [23, 24], [25, 26], [27, 28],
    [13, 29], [30, 31], [32, 33], [34, 35], [36, 37],
    [38, 39], [40, 41], [42, 43], [44, 45], [46, 47],
    [48, 49], [50, 51], [52, 53], [54, 55], [56, 49]
]

# 1514-1536
move_sequence_2= [
    [1, 2], [13, 4], [57, 58], [27, 59], [60, 37]
]

# 1544-1452
move_sequence_3 = [
    [1, 2], [3, 4], [5, 6], [7, 10], [9, 8], [11, 16], [16, 61], [62, 29], [63, 64],
    [65, 66], [67, 68], [69, 70], [25, 22], [27, 71], [72, 73], [74, 31], [13, 75],
    [18, 42], [76, 39], [77, 78], [79, 80], [81, 82], [83, 84], [85, 86], [87, 49],
    [88, 89], [90, 91], [92, 93], [94, 3], [95, 96], [97, 98]
]


chess_games_path = 'filtered_chess_games.json'
chess_games = pd.read_json(chess_games_path)

move_sequence_1 = torch.tensor(move_sequence_1).long().flatten().unsqueeze(0)
move_sequence_2 = torch.tensor(move_sequence_2).long().flatten().unsqueeze(0)
move_sequence_3 = torch.tensor(move_sequence_3).long().flatten().unsqueeze(0)

ratings = chess_games[['white_rating', 'black_rating']].values

mean = np.mean(ratings.flatten())
sd = np.std(ratings.flatten())


y_hat_1 = model(move_sequence_1, model.init_hidden(1))
print("Y HAT 1" ,y_hat_1)
y_hat_1 = mean+(y_hat_1*sd)
print(y_hat_1)

y_hat_2 = model(move_sequence_2, model.init_hidden(1))
y_hat_2 = mean+(y_hat_2*sd)
print(y_hat_2)

y_hat_3 = model(move_sequence_3, model.init_hidden(1))
y_hat_3 = mean+(y_hat_3*sd)
print(y_hat_3)
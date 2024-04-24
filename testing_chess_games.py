import torch
from chess_model import ChessELOPredictor, ChessNextMove
import pandas as pd
import numpy as np
from encode_moves import load_mappings


# Load the trained model
#pretrained_model_path = 'chess_rating_predictor.pt' 
#model = ChessELOPredictor(pretrained_fname=pretrained_model_path, max_moves=6000, hidden_dim=600)
#model = ChessELOPredictor(pretrained_model_path, 6000, 600)

mappings = load_mappings()
model = ChessELOPredictor('nextmove.pt', len(mappings)+1, 64)
model.load_state_dict(torch.load('chess_rating_predictor.pt'))

#model = ChessELOPredictor('chess_rating_predictor.pt' , len(mappings)+1, 64)
model.eval()


# Move sequence where each pair represents a turn
# 1547-1399
move_sequence_1 = [
    [3501, 3505], [3445, 3447], [3503, 3395], [3389, 651], [790, 1268], [707, 957],
    [867, 873], [2304, 3426], [3426, 27], [62, 217], [1718, 1103], [1103, 1165],
    [3618, 719], [870, 1321], [65, 3564], [867, 76], [3677, 2799], [995, 3624],
    [2899, 796], [237, 3480], [2399, 52], [3167, 3376], [3621, 3535], [1124, 386],
    [1425, 1214], [337, 2536], [1767, 1649], [323, 3182], [809, 379], [1309, 2536]
]

# 1514-1536
move_sequence_2 = [
    [3501, 3505],
    [867, 3447],
    [30, 3397],
    [65, 875],
    [3392, 3480]
]

# 1544-1452
move_sequence_3 = [
    [3501, 3505],
    [3445, 3447],
    [3503, 3395],
    [3389, 1268],
    [790, 651],
    [707, 3426],
    [3426, 1084],
    [1211, 76],
    [1723, 49],
    [1491, 252],
    [1764, 1766],
    [442, 805],
    [870, 1165],
    [65, 942],
    [1141, 3601],
    [353, 2799],
    [867, 3681],
    [62, 3621],
    [26, 52],
    [138, 2416],
    [73, 2533],
    [3058, 3297],
    [369, 3340],
    [380, 2194],
    [713, 2536],
    [1105, 2419],
    [2804, 3235],
    [1128, 2200],
    [2781, 3445],
    [1000, 99],
    [2803, 1]
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
y_hat_1 = mean+(y_hat_1*sd)
print(y_hat_1)

y_hat_2 = model(move_sequence_2, model.init_hidden(1))
y_hat_2 = mean+(y_hat_2*sd)
print(y_hat_2)

y_hat_3 = model(move_sequence_3, model.init_hidden(1))
y_hat_3 = mean+(y_hat_3*sd)
print(y_hat_3)
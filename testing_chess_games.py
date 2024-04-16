import torch
from chess_model import ChessRatingPredictor
import joblib

def pad_sequences(sequences, maxlen, value=0):
    # Initialize the padded sequences
    padded_sequences = torch.full((1, maxlen), value)
    seq_length = min(len(sequences), maxlen)
    padded_sequences[0, :seq_length] = torch.tensor(sequences[:seq_length])
    return padded_sequences

# Load the trained model
model = ChessRatingPredictor(move_input_dim=1, hidden_dim=64, output_dim=1)
model.load_state_dict(torch.load('chess_rating_predictor.pth'))
model.eval()


scaler = joblib.load('elo_scaler.pkl')

# Move sequence where each pair represents a turn
# move_sequence = [
#     [3501, 3505], [3445, 3447], [3503, 3395], [3389, 651], [790, 1268], [707, 957],
#     [867, 873], [2304, 3426], [3426, 27], [62, 217], [1718, 1103], [1103, 1165],
#     [3618, 719], [870, 1321], [65, 3564], [867, 76], [3677, 2799], [995, 3624],
#     [2899, 796], [237, 3480], [2399, 52], [3167, 3376], [3621, 3535], [1124, 386],
#     [1425, 1214], [337, 2536], [1767, 1649], [323, 3182], [809, 379], [1309, 2536]
# ]

# move_sequence = [
#     [3501, 3505],
#     [867, 3447],
#     [30, 3397],
#     [65, 875],
#     [3392, 3480]
# ]

# move_sequence = [
#     [3501, 3505],
#     [3445, 3447],
#     [3503, 3395],
#     [3389, 1268],
#     [790, 651],
#     [707, 3426],
#     [3426, 1084],
#     [1211, 76],
#     [1723, 49],
#     [1491, 252],
#     [1764, 1766],
#     [442, 805],
#     [870, 1165],
#     [65, 942],
#     [1141, 3601],
#     [353, 2799],
#     [867, 3681],
#     [62, 3621],
#     [26, 52],
#     [138, 2416],
#     [73, 2533],
#     [3058, 3297],
#     [369, 3340],
#     [380, 2194],
#     [713, 2536],
#     [1105, 2419],
#     [2804, 3235],
#     [1128, 2200],
#     [2781, 3445],
#     [1000, 99],
#     [2803, 1]
# ]

move_sequence = [
    [3501, 3447],
    [3503, 3445],
    [3389, 3564],
    [3426, 1723],
    [867, 1430],
    [83, 3582],
    [3442, 1375],
    [1165, 1500],
    [1103, 1497],
    [3445, 651],
    [1069, 3376],
    [85, 1556],
    [643, 148],
    [107, 875],
    [62, 995],
    [796, 875],
    [1124, 3541],
    [2399, 73],
    [3167, 3561],
    [1209, 1165],
    [1363, 2796],
    [3187, 3430],
    [179, 3558],
    [107, 1494],
    [3656, 2915],
    [128, 1553],
    [1488, 3297],
    [172, 3447],
    [62, 2796],
    [2652, 379],
    [2179, 3679],
    [3337, 3677],
    [3295, 1494],
    [3297, 1306],
    [255, 3621],
    [75, 377],
    [1614, 386],
    [1799, 377],
    [1614, 361],
    [1818, 1256],
    [2781, 1677],
    [91, 2799],
    [1499, 3230],
    [261, 447],
    [222, 1723],
    [128, 1431],
    [2179, 1274],
    [2182, 359],
    [3300, 1218],
    [3677, 376],
    [107, 385],
    [3680, 395],
    [381, 386],
    [382, 1268],
    [3302, 1]
]


# Flatten the sequence
flat_move_sequence = [move for turn in move_sequence for move in turn]

# Assume the maximum sequence length is known (from your training data)
max_sequence_len = 50  # Adjust this based on your actual training data

# Pad the sequence
padded_sequence = pad_sequences(flat_move_sequence, max_sequence_len)

# Convert to tensor and add the necessary dimension for features
input_tensor = torch.tensor(padded_sequence).float().unsqueeze(-1)

# Predict with the model
with torch.no_grad():
    predicted_white_elo, predicted_black_elo = model(input_tensor)
    print(f"Predicted Elo for White: {predicted_white_elo}")
    print(f"Predicted Elo for Black: {predicted_black_elo}")

    # Predict with the model
with torch.no_grad():
    predicted_white_elo, predicted_black_elo = model(input_tensor)
    # Rescale the predicted values
    predicted_elos = scaler.inverse_transform([[predicted_white_elo.item(), predicted_black_elo.item()]])[0]
    print(f"Predicted Elo for White: {predicted_elos[0]}")
    print(f"Predicted Elo for Black: {predicted_elos[1]}")

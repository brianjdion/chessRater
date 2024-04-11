import torch
from prepare_data import ChessRatingPredictor

# Define the model architecture (must match the saved model)
move_input_dim = 1  # Each move is one-dimensional after encoding
rating_input_dim = 2  # Two ratings: white and black
hidden_dim = 64
output_dim = 1  # Predicting one rating at a time, but the model will have two outputs

loaded_model = ChessRatingPredictor(move_input_dim, rating_input_dim, hidden_dim, output_dim)

# Load the saved state dict
loaded_model.load_state_dict(torch.load('chess_rating_predictor.pth'))

# Set the model to evaluation mode
loaded_model.eval()
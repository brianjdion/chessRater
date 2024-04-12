import torch
from model import ChessRatingPredictor  # Import the model class from model.py

# Define the function to load the model
def load_model_state(filename):
    model = ChessRatingPredictor()

    # Load the state dictionary into the model
    model.load_state_dict(torch.load(filename))

    return model

# Load the model state dictionary
model = load_model_state("chessRater.pth")
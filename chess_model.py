import torch
import torch.nn as nn
    
class ChessRatingPredictor(nn.Module):
    def __init__(self, move_input_dim, hidden_dim, output_dim):
        super(ChessRatingPredictor, self).__init__()
        self.lstm = nn.LSTM(move_input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim * 2)  # Output for both white and black ratings

    def forward(self, moves):
        lstm_out, _ = self.lstm(moves)
        lstm_out = lstm_out[:, -1, :]  # Use last hidden state
        ratings = self.fc(lstm_out)
        return ratings[:, 0], ratings[:, 1]

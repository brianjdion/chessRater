import torch
import torch.nn as nn
    
# class ChessRatingPredictor(nn.Module):
#     def __init__(self, move_input_dim, hidden_dim, output_dim):
#         super(ChessRatingPredictor, self).__init__()
#         self.lstm = nn.LSTM(move_input_dim, hidden_dim, batch_first=True, num_layers=3, dropout=0.2)
#         self.dropout = nn.Dropout(0.2)
#         self.fc = nn.Linear(hidden_dim, output_dim * 2)  # Output for both white and black ratings

#     def forward(self, moves):
#         lstm_out, _ = self.lstm(moves)
#         #lstm_out = lstm_out[:, -1, :]  # Use last hidden state
#         lstm_out = self.dropout(lstm_out[:, -1, :])
#         ratings = self.fc(lstm_out)
#         return ratings[:, 0], ratings[:, 1]


class ChessNextMove(nn.Module):
    
    def __init__(self, max_moves, hidden_dim):
        super(ChessNextMove, self).__init__()
        self.nlayers = 3
        self.nhid = hidden_dim
        self.encoder = nn.Embedding(max_moves, self.nhid, padding_idx=0)
        self.lstm = nn.LSTM(self.nhid, self.nhid, batch_first=True, num_layers=self.nlayers, dropout=0.2)
        self.decoder = nn.Linear(self.nhid, max_moves)  # Output for both white and black ratings
        self.decoder.weight = self.encoder.weight
        
    def forward(self, observation, hidden):
        lstm_out, hidden = self.lstm(self.encoder(observation), hidden)
        return self.decoder(lstm_out)

    # def init_hidden(self, bsz):
    #     """ Initialize a fresh hidden state """
    #     weight = next(self.parameters()).data
    #     return (torch.tensor(weight.new(self.nlayers, bsz, self.nhid).zero_()),
    #                 torch.tensor(weight.new(self.nlayers, bsz, self.nhid).zero_()))
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(self.nlayers, bsz, self.nhid).zero_(),
                weight.new(self.nlayers, bsz, self.nhid).zero_())


class ChessELOPredictor(nn.Module):
    
    def __init__(self, pretrained_fname, max_moves, hidden_dim, freeze=True):
        super(ChessELOPredictor, self).__init__()
        self.pretrained = ChessNextMove(max_moves, hidden_dim)
        self.pretrained.load_state_dict(torch.load(pretrained_fname))
        self.nhid = hidden_dim
        self.nlayers = 3

        # Freeze pretrained weights so we only train the decoder 
        if freeze:
            for param in self.pretrained.parameters():
                param.requires_grad = False
        
        self.elo_decoder = nn.Linear(self.nhid, 2)  # Output for both white and black ratings
        
    def forward(self, observation, hidden):
        lstm_out, hidden = self.pretrained.lstm(self.pretrained.encoder(observation), hidden)
        return self.elo_decoder(lstm_out[:,-1,:])

    def init_hidden(self, bsz):
        """ Initialize a fresh hidden state """
        weight = next(self.parameters()).data
        return (torch.tensor(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    torch.tensor(weight.new(self.nlayers, bsz, self.nhid).zero_()))
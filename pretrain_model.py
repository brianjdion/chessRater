import torch
from torch import nn
import pandas as pd
import json
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Declare model:
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

    def init_hidden(self, bsz):
        """ Initialize a fresh hidden state """
        weight = next(self.parameters()).data
        return (torch.tensor(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    torch.tensor(weight.new(self.nlayers, bsz, self.nhid).zero_()))
    
# Encode mapping:
file_path = 'filtered_chess_games.json'
data = pd.read_json(file_path)

def build_mapping(data:pd.DataFrame) -> dict: 
    mappings = {}
    # Reserve 0 for padding
    idx = 1
    for game in data['moves']:
        for pair in game:
            for move in pair: 
                if move not in mappings:
                    mappings[move] = idx
                    idx += 1
    return mappings

def save_mappings(mappings:dict): 
    with open('mapping.json', 'w') as f:
        json.dump(mappings, f)

def load_mappings() -> dict:
    with open('mapping.json', 'r') as f:
        return json.load(f)

def encode_game_moves(game_moves):
    # Encode each pair of moves and collect them into a list
    mappings = load_mappings()
    encoded_game = [[mappings[pair[0]], mappings[pair[1]]] for pair in
                    game_moves]
    return encoded_game

def encode_data(data):
    data['encoded_moves'] = data['moves'].apply(encode_game_moves)
    
    # Extract the encoded moves to a new list
    all_encoded_games = data['encoded_moves'].tolist()
    
    # Save this list to a new JSON file with better readability
    output_file = 'encoded_chess_moves.json'
    with open(output_file, 'w') as f:
        # Use indent and separators for better readability
        json.dump(all_encoded_games, f, indent=1, separators=(',', ': '))


# Load data:
chess_games_path = 'filtered_chess_games.json'  # File with white_rating, black_rating, etc.
encoded_moves_path = 'encoded_chess_moves.json'  # File with encoded move sequences

chess_games = pd.read_json(chess_games_path)
encoded_moves = pd.read_json(encoded_moves_path, typ='series')

chess_games = chess_games.head(10000)
chess_games['encoded_moves'] = encoded_moves.values

# Let's flatten moves and make into input/output 
def flatten_pad_shift(games):
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
    dataset = TensorDataset(torch.tensor(padded[:,:-1]).long(), torch.tensor(padded[:,1:]).long())

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train, test

train, test = flatten_pad_shift(chess_games['encoded_moves'].tolist())
batchSize = 64
train_loader = DataLoader(train, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(test, batch_size=batchSize, shuffle=False)

mappings = load_mappings()
model = ChessNextMove(len(mappings)+1, 600)

epochs = 5
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
cost = torch.nn.CrossEntropyLoss()
for epoch in range(1, epochs+1):
    total_loss = 0
    for batch_idx, (X, Y) in enumerate(train_loader):
        print(batch_idx + 1)
        if (batch_idx+1) % 100 == 0:
            break
        optimizer.zero_grad()
        hidden = model.init_hidden(batchSize)
        logits = model.forward(X, hidden)
        loss = cost(logits.reshape(-1, len(mappings)+1), Y.flatten())
        total_loss += loss.item()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
    print(f'Epoch {epoch}/{epochs}, Total Loss: {total_loss / batch_idx}')

# Save model:
torch.save(model.state_dict(), 'nextmove.pt')
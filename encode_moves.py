import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

# Load the JSON file into a DataFrame
file_path = 'filtered_chess_games.json'
data = pd.read_json(file_path)

# Keep only the first 100 games
data = data.head(10000)

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

# Encode the moves
data['encoded_moves'] = data['moves'].apply(encode_game_moves)
    
# Extract the encoded moves to a new list
all_encoded_games = data['encoded_moves'].tolist()
    
# Save this list to a new JSON file with better readability
output_file = 'encoded_chess_moves.json'
with open(output_file, 'w') as f:
    # Use indent and separators for better readability
    json.dump(all_encoded_games, f, indent=1, separators=(',', ': '))




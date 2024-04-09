import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the JSON file into a DataFrame
file_path = 'filtered_chess_games.json'
data = pd.read_json(file_path)

# Keep only the first 100 games
data = data.head(100)

# Prepare the moves for encoding
all_moves = [move for game in data['moves'] for pair in game for move in pair]

# Initialize and fit the encoder
encoder = LabelEncoder()
encoder.fit(all_moves)

def encode_game_moves(game_moves):
    # Encode each pair of moves and collect them into a list
    encoded_game = [encoder.transform(pair).tolist() for pair in game_moves]
    # Enclose the entire game's moves in a list
    return encoded_game

# Encode the moves
data['encoded_moves'] = data['moves'].apply(encode_game_moves)

# Extract the encoded moves to a new list
all_encoded_games = data['encoded_moves'].tolist()

# Save this list to a new JSON file
output_file = 'encoded_chess_moves.json'
pd.Series(all_encoded_games).to_json(output_file, orient='records', lines=True)

print(f"Encoded moves saved to {output_file}")
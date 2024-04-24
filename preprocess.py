import pandas as pd


## This file path is not used anymore because it's too big of a csv file to be pushed onto GitHub, however we've already
## gotten the information we needed from it that's in filered_chess_games.json
file_path = 'club_games_data 2.csv'  # Replace with the actual file path
data = pd.read_csv(file_path)

# Filter the data for 'blitz' time_class and 'chess' rules only
filtered_data = data[(data['time_class'] == 'blitz') & (data['rules'] == 'chess')]

# Function to extract moves from PGN and organize them into pairs of white and black moves
def extract_moves(pgn_data):
    lines = pgn_data.split('\n')

    # Find the line containing the moves
    moves_line = ''
    for line in lines:
        if line.startswith('1. '):
            moves_line = line
            break

    # Filter out unwanted text and split into individual moves
    moves = moves_line.split(' ')
    moves = [move for move in moves if '.' not in move and '{' not in move and '}' not in move]

    # Pair white and black moves
    paired_moves = [[moves[i], moves[i+1]] for i in range(0, len(moves) - 1, 2)]
    
    return paired_moves

# Extract moves and add them to the DataFrame
filtered_data['moves'] = filtered_data['pgn'].apply(extract_moves)

# Select the required columns
columns_to_include = ['white_rating', 'black_rating', 'white_result', 'black_result', 'moves']
final_data = filtered_data[columns_to_include]

# Convert to JSON and save to a file, ensuring it is in an array format
final_data.to_json('filtered_chess_games.json', orient='records')

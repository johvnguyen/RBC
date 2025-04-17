import random
from reconchess import *
import json
import pprint
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
from preprocessing.RBCDataset import RBCDataset
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def replace_numbers_with_u(strings):
    def replacer(match):
        return 'u' * int(match.group())  # Convert number to integer and repeat 'u' that many times

    return [re.sub(r'\d+', replacer, s) for s in strings]

# Write function to convert FEN string to Tensor
def FEN2InputTensor(fen_string, my_color):
    #my_color = 'white'
    fen_pattern = re.compile(r"^([rnbqkpRNBQKP1-8/]+) ([wb]) ([KQkq-]+) ([a-h1-8-]+) (\d+) (\d+)$")
    if my_color == 'white':
        piece2encoding = {
            'P' : 1,
            'R' : 2,
            'N' : 3,
            'B' : 4,
            'Q' : 5,
            'K' : 6,
        }
    elif my_color == 'black':
        piece2encoding = {
            'p' : 1,
            'r' : 2,
            'n' : 3,
            'b' : 4,
            'q' : 5,
            'k' : 6,
        }

    
    match = fen_pattern.match(fen_string)

    
    #print(f'Input string: {fen_string}')
    
    if match:
        piece_placement, active_color, castling, en_passant, halfmove, fullmove = match.groups()
        #print("Piece Placement:", piece_placement)
        #print("Active Color:", active_color)
        #print("Castling Rights:", castling)
        #print("En Passant Target:", en_passant)
        #print("Halfmove Clock:", halfmove)
        #print("Fullmove Number:", fullmove)

        pieces_by_row = piece_placement.split('/')

        pieces_by_row = replace_numbers_with_u(pieces_by_row)

        # 7 is the value for unknown spaces.
        input_tensor = torch.full((8, 8), 7)

        for (rank, row) in zip(reversed(range(input_tensor.shape[0])), pieces_by_row):  # rows
            for (file, piece) in zip(reversed(range(input_tensor.shape[1])), row):  # columns
                if piece in piece2encoding.keys():
                    input_tensor[rank][file] = piece2encoding[piece]

        #print(f'Input Tensor: {input_tensor}')

        return input_tensor
        
    else:
        print("Invalid FEN string")

def UpdateInputWithSense(input_tensor, my_color, sensing_result):
    my_color = 'white'
    fen_pattern = re.compile(r"^([rnbqkpRNBQKP1-8/]+) ([wb]) ([KQkq-]+) ([a-h1-8-]+) (\d+) (\d+)$")
    if my_color == 'white':
        piece2encoding = {
            'p' : -1,
            'r' : -2,
            'n' : -3,
            'b' : -4,
            'q' : -5,
            'k' : -6,
            'e' : 0,
            'P' : 1,
            'R' : 2,
            'N' : 3,
            'B' : 4,
            'Q' : 5,
            'K' : 6,
        }
    elif my_color == 'black':
        piece2encoding = {
            'P' : -1,
            'R' : -2,
            'N' : -3,
            'B' : -4,
            'Q' : -5,
            'K' : -6,
            'e' : 0,
            'p' : 1,
            'r' : 2,
            'n' : 3,
            'b' : 4,
            'q' : 5,
            'k' : 6,
        }
    for (idx, piece) in sensing_result:
        rank, file = divmod(idx, 8)

        if piece != None:
            input_tensor[rank][file] = piece2encoding[piece.symbol()]
        else:
            input_tensor[rank][file] = piece2encoding['e']

    return input_tensor

def replace_numbers_with_e(strings):
    def replacer(match):
        return 'e' * int(match.group())  # Convert number to integer and repeat 'e' that many times

    return [re.sub(r'\d+', replacer, s) for s in strings]

# Write function to convert FEN string to Output Tensor
def FEN2OutputTensor(fen_string, my_color):
    #my_color = 'white'
    fen_pattern = re.compile(r"^([rnbqkpRNBQKP1-8/]+) ([wb]) ([KQkq-]+) ([a-h1-8-]+) (\d+) (\d+)$")
    if my_color == 'white':
        opp_piece2channel_idx = {
            'e' : 0,
            'P' : 1,
            'R' : 2,
            'N' : 3,
            'B' : 4,
            'Q' : 5,
            'K' : 6,
            'p' : -1,
            'r' : -2,
            'n' : -3,
            'b' : -4,
            'q' : -5,
            'k' : -6
        }
    elif my_color == 'black':
        opp_piece2channel_idx = {
            'e' : 0,   # Empty space
            'p' : 1,   # My pawn
            'r' : 2,   # My rook
            'n' : 3,   # My knight
            'b' : 4,   # My bishop
            'q' : 5,   # My queen
            'k' : 6,   # My king
            'P' : 7,   # Opp pawn
            'R' : 8,   # Opp rook
            'N' : 9,   # Opp knight
            'B' : 10,  # Opp bishop
            'Q' : 11,  # Opp queen
            'K' : 12   # Opp king
        }



    
    match = fen_pattern.match(fen_string)
    
    if match:
        piece_placement, active_color, castling, en_passant, halfmove, fullmove = match.groups()
        #print("Piece Placement:", piece_placement)
        #print("Active Color:", active_color)
        #print("Castling Rights:", castling)
        #print("En Passant Target:", en_passant)
        #print("Halfmove Clock:", halfmove)
        #print("Fullmove Number:", fullmove)

        pieces_by_row = piece_placement.split('/')

        pieces_by_row = replace_numbers_with_e(pieces_by_row)
        #print(f'modified fen string: {pieces_by_row}')

        output_tensor = torch.zeros((8, 8, 6 + 6 + 1))

        for (rank, fen_row) in zip(reversed(range(output_tensor.shape[0])), pieces_by_row):
            for (file, piece) in zip(reversed(range(output_tensor.shape[1])), fen_row):
                output_tensor[rank][file][opp_piece2channel_idx[piece]] = 1
        return output_tensor
        
    else:
        print("Invalid FEN string")

def ensure_file_path(filepath: str) -> bool:
    path = Path(filepath)
    
    if path.is_file():
        return True
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    return False

def add_to_history(input_history, new_input, max_seq_len = 100):
    input_history.append(new_input)

    if len(input_history) == 101:
        input_history = input_history[:-1]

    return

def filter_game(rbc_game):
    # WinReason == 2 <-> Timeout
    if rbc_game.get_win_reason() == 2:
        return True
    return False

max_seq_len = 100


def Game2Dataset(fpath, max_seq_len = 100):
    if os.path.isfile(fpath):
        print("File exists.")
    else:
        raise IOError("File does not exist.")
        return
    
    game_history = GameHistory.from_file(fpath)
    fname = fname = os.path.splitext(os.path.basename(fpath))[0]
    input_history = []
    input_seqs = []
    outputs = []

    if filter_game(game_history):
        return
    
    
    nturns = game_history.num_turns()
    turns = list(game_history.turns())
    
    for i in range(len(turns)):
        turn = turns[i]
        if i % 2 == 0:
            turn_player = "white"
        else:
            turn_player = "black"
        
        postsense_input = None
    
        # Ture boardstate pre-sense
        true_boardstate = game_history.truth_fen_before_move(turn)
        output = FEN2OutputTensor(true_boardstate, my_color = turn_player)
    
        # Known boardstate pre-sense
        presense_input = FEN2InputTensor(true_boardstate, my_color = turn_player)
    
        
        
        # Sensing action?
        if game_history.has_sense(turn):
            sensing_result = game_history.sense_result(turn)
            postsense_input = UpdateInputWithSense(presense_input, my_color = turn_player, sensing_result = sensing_result)
    
        add_to_history(input_history, presense_input, max_seq_len = max_seq_len)
        
        input_seqs.append(torch.stack(input_history, dim = 0))
        outputs.append(output)
    
        if postsense_input is not None:
            add_to_history(input_history, postsense_input, max_seq_len = max_seq_len)
            input_seqs.append(torch.stack(input_history, dim = 0))
            outputs.append(output)
    
        dataset = RBCDataset(input_seqs, outputs)
        ensure_file_path(f'{fname}.pth')
        torch.save(dataset, f"data/datasets/{fname}.pth")


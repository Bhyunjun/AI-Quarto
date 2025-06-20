import numpy as np
import random
from itertools import product
from mcts import MCTS
from mcts import MCTSNode
from mcts import QuartoState
import time

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
        self.next_move = None
        self.next_state = None
        self.next_piece = None


    def select_piece(self):
        # Available locations to place the piece
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]
        # Make your own algorithm here
        if len(self.available_pieces) > 13:
            return random.choice(self.available_pieces)

        mcts = MCTS(exploration_weight=1.0)
        initial_state = QuartoState(board=self.board, remaining_pieces=self.available_pieces,
                                    current_piece=None)

        self.next_state = mcts.search(initial_state, itermax=40000, max_depth=8, time_limit=20.0)

        return self.next_state.last_piece



    def place_piece(self, selected_piece):
        # Available locations to place the piece
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]
        # Make your own algorithm here
        if len(available_locs) > 13:
            return random.choice(available_locs)

        mcts = MCTS(exploration_weight=1.0)
        initial_state = QuartoState(board=self.board, remaining_pieces=self.available_pieces,
                                    current_piece=selected_piece, turn = 1, inturn=1)

        self.next_state = mcts.search(initial_state, itermax=40000, max_depth=8, time_limit=20.0)
        # time.sleep(1) # Check time consumption (Delete when you make your algorithm)

        return self.next_state.last_move
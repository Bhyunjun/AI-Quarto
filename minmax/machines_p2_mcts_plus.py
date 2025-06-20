import numpy as np
import random
from itertools import product
from mcts_plus import MCTSPlus, MCTSNode, QuartoState
import time

class P2():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces
        self.next_move = None
        self.next_state = None
        self.next_piece = None
    
    def select_piece(self):
        # 초반에는 빠른 선택 (13개 이상 남았을 때)
        if len(self.available_pieces) > 13:
            return random.choice(self.available_pieces)

        # MCTS+ 엔진 사용 (P2는 약간 더 보수적)
        mcts_plus = MCTSPlus(exploration_weight=1.2)  # 더 탐험적
        initial_state = QuartoState(board=self.board, remaining_pieces=self.available_pieces,
                                  current_piece=None)

        # 강화된 파라미터: 더 많은 반복수와 더 긴 시간
        self.next_state = mcts_plus.search(initial_state, itermax=30000, max_depth=10, time_limit=22.0)

        return self.next_state.last_piece

    def place_piece(self, selected_piece):
        # 초반에는 빠른 배치 (13개 이상 남았을 때)
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]
        if len(available_locs) > 13:
            return random.choice(available_locs)

        # MCTS+ 엔진 사용 (P2는 약간 더 보수적)
        mcts_plus = MCTSPlus(exploration_weight=1.2)  # 더 탐험적
        initial_state = QuartoState(board=self.board, remaining_pieces=self.available_pieces,
                                  current_piece=selected_piece, turn=1, inturn=1)

        # 강화된 파라미터: 더 많은 반복수와 더 긴 시간
        self.next_state = mcts_plus.search(initial_state, itermax=30000, max_depth=10, time_limit=22.0)

        return self.next_state.last_move
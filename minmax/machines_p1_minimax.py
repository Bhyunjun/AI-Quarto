import numpy as np
import random
from itertools import product
import time

# Minimax 전략 함수들 import
from strategy_minimax import (
    minimax_select_piece,
    minimax_place_piece,
    find_winning_moves,
    get_empty_positions
)

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
        
    def select_piece(self):
        """P1의 말 선택 - 기회 포착형 (계산된 압박)"""
        try:
            # P1 특성: 약간 공격적으로 상대방 압박
            selected = minimax_select_piece(self.board, self.available_pieces, player_id=1, is_aggressive=True)
            return selected if selected else random.choice(self.available_pieces)
        except Exception as e:
            print(f"P1 select_piece error: {e}")
            return random.choice(self.available_pieces) if self.available_pieces else None

    def place_piece(self, selected_piece):
        """P1의 말 배치 - 적극적 승리 추구"""
        available_locs = get_empty_positions(self.board)
        
        if not available_locs:
            return None
        
        try:
            # P1 특성: 공격적으로 승리 기회 창출
            best_move = minimax_place_piece(self.board, selected_piece, self.available_pieces, 
                                          player_id=1, is_aggressive=True)
            
            if best_move and best_move in available_locs:
                return best_move
            else:
                return random.choice(available_locs)
                
        except Exception as e:
            print(f"P1 place_piece error: {e}")
            return random.choice(available_locs)
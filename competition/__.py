import numpy as np
import random
from itertools import product
import time

class P2():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces

    def select_piece(self):
        def evaluate_piece_danger(piece):
            idx = self.pieces.index(piece) + 1
            danger = 0

            for r in range(4):
                for c in range(4):
                    if self.board[r][c] == 0:
                        temp_board = self.board.copy()
                        temp_board[r, c] = idx

                        if self.check_win(temp_board):
                            return 1000  # 즉시 승리 가능: 매우 위험

                        danger += self.evaluate_board_threat(temp_board)

            return danger

        def get_danger_score(item):
            return item[1]

        piece_scores = [(piece, evaluate_piece_danger(piece)) for piece in self.available_pieces]
        piece_scores.sort(key=get_danger_score)  # lambda 없이 함수로 정렬
        return piece_scores[0][0]

    def place_piece(self, selected_piece):
        piece_idx = self.pieces.index(selected_piece) + 1
        available_locs = [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]

        for r, c in available_locs:
            temp_board = self.board.copy()
            temp_board[r, c] = piece_idx
            if self.check_win(temp_board):
                return (r, c)

        if (1, 1) in available_locs:
            return (1, 1)
        for pos in [(0, 0), (0, 3), (3, 0), (3, 3)]:
            if pos in available_locs:
                return pos

        return random.choice(available_locs)

    def check_win(self, board):
        def check_line(line):
            if 0 in line:
                return False
            values = np.array([self.pieces[idx - 1] for idx in line])
            for i in range(4):
                if len(set(values[:, i])) == 1:
                    return True
            return False

        for i in range(4):
            if check_line([board[i, j] for j in range(4)]):
                return True
            if check_line([board[j, i] for j in range(4)]):
                return True

        if check_line([board[i, i] for i in range(4)]) or check_line([board[i, 3 - i] for i in range(4)]):
            return True

        for r in range(3):
            for c in range(3):
                group = [board[r, c], board[r+1, c], board[r, c+1], board[r+1, c+1]]
                if 0 in group:
                    continue
                values = [self.pieces[idx - 1] for idx in group]
                for i in range(4):
                    if len(set([v[i] for v in values])) == 1:
                        return True

        return False

    def evaluate_board_threat(self, board):
        threat = 0
        lines = []

        for i in range(4):
            lines.append([board[i, j] for j in range(4)])
            lines.append([board[j, i] for j in range(4)])
        lines.append([board[i, i] for i in range(4)])
        lines.append([board[i, 3 - i] for i in range(4)])
        for r in range(3):
            for c in range(3):
                lines.append([board[r, c], board[r+1, c], board[r, c+1], board[r+1, c+1]])

        for line in lines:
            nonzero = [idx for idx in line if idx != 0]
            if len(nonzero) == 3:
                values = [self.pieces[idx - 1] for idx in nonzero]
                for i in range(4):
                    if len(set([v[i] for v in values])) == 1:
                        threat += 10

        return threat
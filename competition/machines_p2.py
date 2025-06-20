# machines_p1.py  (Full-Option Attack P1)
# Python 3.12  NumPy ≥ 1.24
import numpy as np
import random

class P2:
    """
    강화형 P1
    - place_piece : 즉승 → 위험 최소 위치
    - select_piece: 상대 즉승 가능성 최소 조각
    """

    def __init__(self, board, available_pieces):
        self.board = board                    # numpy 4×4
        self.available_pieces = available_pieces
        # 16 개의 MBTI 조각 (고정 인덱스 유지)
        self.pieces = [(i, j, k, l)
                       for i in range(2)
                       for j in range(2)
                       for k in range(2)
                       for l in range(2)]

    # ---------------------------------------------------
    # 1) 상대에게 줄 조각 고르기
    # ---------------------------------------------------
    def select_piece(self):
        best_candidates = []
        min_risk = float("inf")

        for piece in self.available_pieces:
            risk = self._immediate_win_risk(piece)
            if risk < min_risk:
                min_risk = risk
                best_candidates = [piece]
            elif risk == min_risk:
                best_candidates.append(piece)

        # 위험이 동일하면 랜덤으로
        return random.choice(best_candidates)

    # ---------------------------------------------------
    # 2) 받은 조각 놓을 위치 고르기
    # ---------------------------------------------------
    def place_piece(self, selected_piece):
        piece_idx = self.pieces.index(selected_piece) + 1
        avail_cells = [(r, c) for r in range(4) for c in range(4)
                       if self.board[r][c] == 0]

        # (1) 즉시 이길 수 있으면 바로 승리
        for r, c in avail_cells:
            tmp = self.board.copy()
            tmp[r, c] = piece_idx
            if self._check_win(tmp):
                return (r, c)

        # (2) 각 위치별 “상대 즉승 가능 위험도” 계산
        safest = []
        min_risk = float("inf")

        remaining_pieces = [p for p in self.available_pieces
                            if p != selected_piece]

        for r, c in avail_cells:
            tmp = self.board.copy()
            tmp[r, c] = piece_idx
            risk = 0
            # 상대가 줄 수 있는 모든 조각 시뮬레이션
            for opp_piece in remaining_pieces:
                idx = self.pieces.index(opp_piece) + 1
                for rr, cc in [(x, y) for x in range(4) for y in range(4)
                               if tmp[x, y] == 0]:
                    tmp2 = tmp.copy()
                    tmp2[rr, cc] = idx
                    if self._check_win(tmp2):
                        risk += 1
                        break       # 이 조각은 위험, 더 볼 필요 없음
            if risk < min_risk:
                min_risk = risk
                safest = [(r, c)]
            elif risk == min_risk:
                safest.append((r, c))

        # 위험이 가장 낮은 위치 중 랜덤 선택
        return random.choice(safest)

    # ---------------------------------------------------
    # 내부 유틸
    # ---------------------------------------------------
    def _immediate_win_risk(self, piece):
        """상대에게 piece를 줬을 때 즉시 승리할 수 있는 칸 개수"""
        idx = self.pieces.index(piece) + 1
        risk = 0
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:
                    tmp = self.board.copy()
                    tmp[r, c] = idx
                    if self._check_win(tmp):
                        risk += 1
                        break
        return risk


    def _check_win(self, board):
        """P2 와 동일한 승리 판정 (행·열·대각 + 2×2 서브그리드)"""
        def same_trait(line):
            if 0 in line:
                return False
            vals = np.array([self.pieces[i - 1] for i in line])
            for k in range(4):               # I/E, N/S, T/F, P/J
                if len(set(vals[:, k])) == 1:
                    return True
            return False

        # 행·열
        for i in range(4):
            if same_trait([board[i, j] for j in range(4)]):         return True
            if same_trait([board[j, i] for j in range(4)]):         return True
        # 대각
        if same_trait([board[i, i] for i in range(4)]):             return True
        if same_trait([board[i, 3 - i] for i in range(4)]):         return True
        # 2×2 서브그리드
        for r in range(3):
            for c in range(3):
                group = [board[r, c], board[r+1, c],
                         board[r, c+1], board[r+1, c+1]]
                if 0 in group:
                    continue
                vals = [self.pieces[g - 1] for g in group]
                for k in range(4):
                    if len(set(v[k] for v in vals)) == 1:
                        return True
        return False

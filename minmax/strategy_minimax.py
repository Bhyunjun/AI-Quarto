import numpy as np
import random
import math
import time
from copy import deepcopy
from functools import lru_cache

# ==================== 기존 호환성 ====================
ALL_PIECES = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]

# 보드 위치 그룹
CENTER_POSITIONS = [(1,1), (1,2), (2,1), (2,2)]
CORNER_POSITIONS = [(0,0), (0,3), (3,0), (3,3)]
EDGE_POSITIONS = [(0,1), (0,2), (1,0), (1,3), (2,0), (2,3), (3,1), (3,2)]

# 승리 라인들
WIN_LINES = []
for row in range(4):
    WIN_LINES.append([(row, col) for col in range(4)])
for col in range(4):
    WIN_LINES.append([(row, col) for row in range(4)])
WIN_LINES.append([(i, i) for i in range(4)])
WIN_LINES.append([(i, 3-i) for i in range(4)])
for r in range(3):
    for c in range(3):
        WIN_LINES.append([(r, c), (r, c+1), (r+1, c), (r+1, c+1)])

# ==================== 게임 로직 최적화 ====================
@lru_cache(maxsize=8192)
def check_line_win_cached(line_tuple):
    """캐시된 라인 승리 체크"""
    if 0 in line_tuple:
        return False
    
    try:
        pieces = [ALL_PIECES[idx - 1] for idx in line_tuple]
        for attr_idx in range(4):
            if len(set(piece[attr_idx] for piece in pieces)) == 1:
                return True
    except (IndexError, TypeError):
        pass
    return False

def check_board_win_fast(board):
    """최적화된 승리 체크"""
    for line_positions in WIN_LINES:
        line_tuple = tuple(board[r, c] for r, c in line_positions)
        if check_line_win_cached(line_tuple):
            return True
    return False

def get_board_hash(board):
    """보드 해시 생성 (전치표용)"""
    return hash(board.tobytes())

def get_empty_positions(board):
    """빈 위치들 반환"""
    return [(r, c) for r in range(4) for c in range(4) if board[r, c] == 0]

def can_win_immediately(board, piece, position):
    """즉시 승리 체크"""
    try:
        temp_board = board.copy()
        piece_idx = ALL_PIECES.index(piece) + 1
        temp_board[position[0], position[1]] = piece_idx
        return check_board_win_fast(temp_board)
    except:
        return False

def find_winning_moves(board, piece):
    """즉시 승리 가능한 모든 위치"""
    return [pos for pos in get_empty_positions(board) 
            if can_win_immediately(board, piece, pos)]

# ==================== 고급 평가 함수 ====================
class AdvancedEvaluator:
    @staticmethod
    def evaluate_board(board, available_pieces, maximizing_player):
        """향상된 보드 평가 함수"""
        if check_board_win_fast(board):
            return 10000 if maximizing_player else -10000
        
        if np.all(board != 0):  # 보드 가득참
            return 0
        
        score = 0
        
        # 1. 즉시 위협 평가 (매우 높은 가중치)
        score += AdvancedEvaluator._evaluate_immediate_threats(board, available_pieces) * 0.5
        
        # 2. 다음 턴 위협 평가
        score += AdvancedEvaluator._evaluate_next_turn_threats(board, available_pieces) * 0.3
        
        # 3. 위치 지배력 평가
        score += AdvancedEvaluator._evaluate_positional_strength(board) * 0.15
        
        # 4. 말 다양성 평가
        score += AdvancedEvaluator._evaluate_piece_diversity(available_pieces) * 0.05
        
        return score if maximizing_player else -score
    
    @staticmethod
    def _evaluate_immediate_threats(board, available_pieces):
        """즉시 위협 상황 평가"""
        threat_score = 0
        
        for piece in available_pieces:
            winning_moves = find_winning_moves(board, piece)
            if winning_moves:
                # 즉시 승리 가능한 말은 매우 위험
                threat_score += len(winning_moves) * 1000
        
        return threat_score
    
    @staticmethod
    def _evaluate_next_turn_threats(board, available_pieces):
        """다음 턴 위협 평가"""
        threat_score = 0
        
        for line_positions in WIN_LINES:
            line_values = [board[r, c] for r, c in line_positions]
            empty_count = line_values.count(0)
            filled_pieces = [ALL_PIECES[idx - 1] for idx in line_values if idx != 0]
            
            if len(filled_pieces) == 3 and empty_count == 1:
                # 3개 채워진 라인
                for attr_idx in range(4):
                    try:
                        attrs = [piece[attr_idx] for piece in filled_pieces]
                        if len(set(attrs)) == 1:
                            threat_score += 500  # 매우 위험
                            break
                    except:
                        continue
                        
            elif len(filled_pieces) == 2 and empty_count == 2:
                # 2개 채워진 라인
                for attr_idx in range(4):
                    try:
                        attrs = [piece[attr_idx] for piece in filled_pieces]
                        if len(set(attrs)) == 1:
                            threat_score += 100  # 중간 위험
                            break
                    except:
                        continue
        
        return threat_score
    
    @staticmethod
    def _evaluate_positional_strength(board):
        """위치 강도 평가"""
        position_score = 0
        
        # 중앙 제어 보너스
        center_control = sum(1 for pos in CENTER_POSITIONS 
                           if board[pos[0], pos[1]] != 0)
        position_score += center_control * 20
        
        # 코너 제어 보너스
        corner_control = sum(1 for pos in CORNER_POSITIONS 
                           if board[pos[0], pos[1]] != 0)
        position_score += corner_control * 10
        
        # 라인 참여도
        lines_with_pieces = 0
        for line_positions in WIN_LINES:
            if any(board[r, c] != 0 for r, c in line_positions):
                lines_with_pieces += 1
        
        position_score += lines_with_pieces * 2
        
        return position_score
    
    @staticmethod
    def _evaluate_piece_diversity(available_pieces):
        """말 다양성 평가"""
        if not available_pieces:
            return 0
        
        # 균형잡힌 말들의 비율
        balanced_count = 0
        for piece in available_pieces:
            if piece.count(0) == 2 and piece.count(1) == 2:
                balanced_count += 1
        
        return (balanced_count / len(available_pieces)) * 20

# ==================== 강화된 Minimax 엔진 ====================
class EnhancedMinimaxEngine:
    def __init__(self, max_time=100):
        self.max_time = max_time
        self.transposition_table = {}
        self.nodes_searched = 0
        self.start_time = 0
        
    def clear_cache(self):
        """캐시 정리"""
        self.transposition_table.clear()
        check_line_win_cached.cache_clear()
        
    def search(self, board, available_pieces, selected_piece, max_depth=8):
        """반복적 심화 탐색"""
        self.start_time = time.time()
        self.nodes_searched = 0
        
        best_move = None
        best_value = float('-inf')
        
        # 반복적 심화: 깊이 1부터 max_depth까지
        for depth in range(1, max_depth + 1):
            if self._out_of_time():
                break
                
            try:
                move, value = self._minimax(
                    board, available_pieces, selected_piece, 
                    depth, True, float('-inf'), float('inf')
                )
                
                if move is not None:
                    best_move = move
                    best_value = value
                    
            except TimeoutError:
                break
        
        return best_move, best_value
    
    def _minimax(self, board, available_pieces, selected_piece, depth, maximizing, alpha, beta):
        """강화된 미니맥스"""
        if self._out_of_time():
            raise TimeoutError()
            
        self.nodes_searched += 1
        
        # 전치표 체크
        state_key = self._get_state_key(board, available_pieces, selected_piece, maximizing, depth)
        if state_key in self.transposition_table:
            cached_depth, cached_value = self.transposition_table[state_key]
            if cached_depth >= depth:
                return None, cached_value
        
        # 기저 조건
        if check_board_win_fast(board):
            value = 10000 if maximizing else -10000
            self.transposition_table[state_key] = (depth, value)
            return None, value
        
        if depth == 0 or np.all(board != 0) or not available_pieces:
            value = AdvancedEvaluator.evaluate_board(board, available_pieces, maximizing)
            self.transposition_table[state_key] = (depth, value)
            return None, value
        
        if selected_piece is None:
            # 말 선택 단계
            return self._search_piece_selection(board, available_pieces, depth, maximizing, alpha, beta)
        else:
            # 말 배치 단계
            return self._search_piece_placement(board, available_pieces, selected_piece, depth, maximizing, alpha, beta)
    
    def _search_piece_selection(self, board, available_pieces, depth, maximizing, alpha, beta):
        """말 선택 탐색"""
        best_piece = None
        
        if maximizing:
            max_eval = float('-inf')
            
            # 이동 순서: 안전한 말 우선
            safe_pieces = [p for p in available_pieces if not find_winning_moves(board, p)]
            risky_pieces = [p for p in available_pieces if p not in safe_pieces]
            ordered_pieces = safe_pieces + risky_pieces
            
            for piece in ordered_pieces:
                if self._out_of_time():
                    break
                    
                _, eval_score = self._minimax(
                    board, available_pieces, piece, depth, False, alpha, beta
                )
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_piece = piece
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return best_piece, max_eval
        else:
            min_eval = float('inf')
            
            # 상대방 관점에서 좋은 말 순서
            safe_pieces = [p for p in available_pieces if not find_winning_moves(board, p)]
            risky_pieces = [p for p in available_pieces if p not in safe_pieces]
            ordered_pieces = risky_pieces + safe_pieces  # 상대방은 위험한 말 우선
            
            for piece in ordered_pieces:
                if self._out_of_time():
                    break
                    
                _, eval_score = self._minimax(
                    board, available_pieces, piece, depth, True, alpha, beta
                )
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_piece = piece
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return best_piece, min_eval
    
    def _search_piece_placement(self, board, available_pieces, selected_piece, depth, maximizing, alpha, beta):
        """말 배치 탐색"""
        empty_positions = get_empty_positions(board)
        best_position = None
        
        if maximizing:
            max_eval = float('-inf')
            
            # 이동 순서: 승리 > 중앙 > 코너 > 기타
            winning_moves = find_winning_moves(board, selected_piece)
            if winning_moves:
                return winning_moves[0], 10000
            
            ordered_positions = self._order_positions(empty_positions)
            
            for pos in ordered_positions:
                if self._out_of_time():
                    break
                
                new_board = board.copy()
                piece_idx = ALL_PIECES.index(selected_piece) + 1
                new_board[pos[0], pos[1]] = piece_idx
                new_available = [p for p in available_pieces if p != selected_piece]
                
                _, eval_score = self._minimax(
                    new_board, new_available, None, depth - 1, True, alpha, beta
                )
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_position = pos
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return best_position, max_eval
        else:
            min_eval = float('inf')
            
            # 즉시 승리 체크
            winning_moves = find_winning_moves(board, selected_piece)
            if winning_moves:
                return winning_moves[0], -10000
            
            ordered_positions = self._order_positions(empty_positions)
            
            for pos in ordered_positions:
                if self._out_of_time():
                    break
                
                new_board = board.copy()
                piece_idx = ALL_PIECES.index(selected_piece) + 1
                new_board[pos[0], pos[1]] = piece_idx
                new_available = [p for p in available_pieces if p != selected_piece]
                
                _, eval_score = self._minimax(
                    new_board, new_available, None, depth - 1, False, alpha, beta
                )
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_position = pos
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return best_position, min_eval
    
    def _order_positions(self, positions):
        """위치 순서 최적화"""
        def position_priority(pos):
            if pos in CENTER_POSITIONS:
                return 0  # 최우선
            elif pos in CORNER_POSITIONS:
                return 1  # 두 번째
            else:
                return 2  # 마지막
        
        return sorted(positions, key=position_priority)
    
    def _get_state_key(self, board, available_pieces, selected_piece, maximizing, depth):
        """상태 키 생성"""
        board_hash = get_board_hash(board)
        pieces_hash = hash(tuple(sorted(available_pieces)))
        piece_hash = hash(selected_piece) if selected_piece else 0
        return (board_hash, pieces_hash, piece_hash, maximizing, depth)
    
    def _out_of_time(self):
        """시간 초과 체크"""
        return (time.time() - self.start_time) >= self.max_time

# ==================== 메인 인터페이스 ====================
def enhanced_minimax_select_piece(board, available_pieces, player_id, is_aggressive=True):
    """강화된 Minimax 말 선택"""
    if not available_pieces:
        return None
    
    try:
        # 즉시 패배 방지
        safe_pieces = [p for p in available_pieces if not find_winning_moves(board, p)]
        search_pieces = safe_pieces if safe_pieces else available_pieces
        
        # 시간 배분
        pieces_on_board = np.sum(board != 0)
        if pieces_on_board <= 4:
            max_time = 20  # 오프닝
            max_depth = 8
        elif pieces_on_board <= 10:
            max_time = 36  # 미들게임
            max_depth = 12
        else:
            max_time = 80  # 엔드게임
            max_depth = 30
        
        engine = EnhancedMinimaxEngine(max_time)
        best_piece, value = engine.search(board, search_pieces, None, max_depth)
        
        if best_piece and best_piece in search_pieces:
            return best_piece
        
    except Exception as e:
        print(f"Enhanced minimax select error: {e}")
    
    # 폴백
    safe_pieces = [p for p in available_pieces if not find_winning_moves(board, p)]
    return random.choice(safe_pieces if safe_pieces else available_pieces)

def enhanced_minimax_place_piece(board, selected_piece, available_pieces, player_id, is_aggressive=True):
    """강화된 Minimax 말 배치"""
    empty_positions = get_empty_positions(board)
    
    if not empty_positions:
        return None
    
    # 즉시 승리 체크
    winning_moves = find_winning_moves(board, selected_piece)
    if winning_moves:
        return winning_moves[0]
    
    try:
        # 시간 배분
        pieces_on_board = np.sum(board != 0)
        if pieces_on_board <= 4:
            max_time = 24  # 오프닝
            max_depth = 8
        elif pieces_on_board <= 10:
            max_time = 45  # 미들게임
            max_depth = 12
        else:
            max_time = 90  # 엔드게임
            max_depth = 30
        
        engine = EnhancedMinimaxEngine(max_time)
        best_position, value = engine.search(board, available_pieces, selected_piece, max_depth)
        
        if best_position and best_position in empty_positions:
            return best_position
        
    except Exception as e:
        print(f"Enhanced minimax place error: {e}")
    
    # 폴백: 전략적 위치
    for pos in CENTER_POSITIONS:
        if pos in empty_positions:
            return pos
    
    for pos in CORNER_POSITIONS:
        if pos in empty_positions:
            return pos
    
    return random.choice(empty_positions)

# ==================== 호환성을 위한 별칭 ====================
minimax_select_piece = enhanced_minimax_select_piece
minimax_place_piece = enhanced_minimax_place_piece
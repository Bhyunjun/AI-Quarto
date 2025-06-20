import math
import random
import numpy as np
from copy import deepcopy
import time

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.action = action

    def is_fully_expanded(self):
        if self.state.current_piece is None:
            return len(self.children) == len(self.state.remaining_pieces)
        else:
            return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, exploration_weight=1.0):
        if not self.children:
            raise ValueError("No children available to select the best child.")
        choices_weights = []
        for child in self.children:
            if child.visits > 0:
                exploitation = child.value / child.visits
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                choices_weights.append(exploitation + exploration)
            else:
                choices_weights.append(float('inf'))
        return self.children[choices_weights.index(max(choices_weights))]

class MCTSPlus:
    def __init__(self, exploration_weight=1.0):
        self.exploration_weight = exploration_weight

    def search(self, initial_state, itermax=10000, max_depth=10, time_limit=22.0):
        """
        성능 우선 MCTS+:
        - 높은 반복수 (10K회) + 22초 제한
        - 안전 장치 제거로 온전한 22초 활용
        - 기본 MCTS와 동일한 신뢰성 있는 시간 관리
        """
        # 간소화된 적응적 파라미터 (오버헤드 최소화)
        adaptive_time, adaptive_iter = self._calculate_simple_params(initial_state, time_limit, itermax)
        
        root = MCTSNode(state=initial_state)
        start_time = time.time()
        iterations = 0

        print(f"MCTS+ 시작: {adaptive_iter}회 반복, {adaptive_time:.1f}초 제한")

        # 기본 MCTS와 동일한 간단한 루프 (안전 장치 제거)
        while iterations < adaptive_iter and (time.time() - start_time) < adaptive_time:
            node = self._select(root)
            if not node.state.is_terminal():
                # 성능 향상된 시뮬레이션 (8% 스마트)
                reward = self._simplified_simulate(node.state, max_depth=max_depth)
                self._backpropagate(node, reward)
            else:
                self._backpropagate(node, node.state.get_result())
            iterations += 1

        if not root.children:
            raise ValueError("Search completed but no children were expanded.")

        elapsed_time = time.time() - start_time
        print(f"MCTS+ 완료: {iterations}회 반복, {elapsed_time:.1f}초 소요")

        # 간소화된 best_child 선택 (출력 최소화)
        if len(root.children) <= 5:
            print("==== MCTS+ Selection ====")
            for child in root.children:
                win_rate = child.value/child.visits if child.visits > 0 else 0
                print(f"Action: {child.action}, Visits: {child.visits}, WinRate: {win_rate:.3f}")

        best_child = root.best_child(0)
        print(f"Selected: {best_child.action}")
        print("=" * 25)
        return best_child.state

    def _calculate_simple_params(self, state, base_time_limit, base_itermax):
        """간소화된 적응적 파라미터 (오버헤드 최소화)"""
        pieces_on_board = np.sum(state.board != 0)
        
        # 간단한 3단계 조정만
        if pieces_on_board <= 6:  # 오프닝
            time_multiplier = 0.7  
            iter_multiplier = 0.8
        elif pieces_on_board <= 12:  # 미들게임
            time_multiplier = 1.0
            iter_multiplier = 1.0
        else:  # 엔드게임
            time_multiplier = 1.2
            iter_multiplier = 1.1
        
        # 위험 상황 감지 간소화 (샘플링 축소)
        threat_count = self._count_threats_fast(state)
        if threat_count >= 2:
            time_multiplier *= 1.1  # 기존 1.2 → 1.1로 완화
            iter_multiplier *= 1.1
        
        adaptive_time = min(base_time_limit * time_multiplier, 24.0)  # 최대 24초 (성능 우선)
        adaptive_iter = int(base_itermax * iter_multiplier)
        
        return adaptive_time, adaptive_iter

    def _count_threats_fast(self, state):
        """빠른 위험 상황 체크 (샘플링 축소)"""
        threat_count = 0
        # 기존 8개 → 4개로 샘플링 축소
        for piece in state.remaining_pieces[:4]:
            if self._has_winning_move_fast(state, piece):
                threat_count += 1
        return threat_count

    def _has_winning_move_fast(self, state, piece):
        """빠른 승리 가능성 체크 (샘플링 축소)"""
        if piece not in state.remaining_pieces:
            return False
        
        legal_moves = state.get_legal_moves()
        # 기존 6개 → 3개로 샘플링 축소
        for move in legal_moves[:3]:
            try:
                temp_state = deepcopy(state)
                temp_state.current_piece = piece
                temp_state.make_move(move)
                if temp_state.check_victory():
                    return True
            except:
                continue
        return False

    def _select(self, node):
        while not node.state.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        if not node.state.is_terminal():
            return self._expand(node)
        return node

    def _expand(self, node):
        if node.state.current_piece is None:
            tried_pieces = [child.action for child in node.children]
            for piece in node.state.remaining_pieces:
                if piece not in tried_pieces:
                    new_state = deepcopy(node.state)
                    new_state.make_move(piece)
                    child_node = MCTSNode(state=new_state, parent=node, action=piece)
                    node.children.append(child_node)
                    return child_node
        else:
            tried_moves = [child.action for child in node.children]
            for move in node.state.get_legal_moves():
                if move not in tried_moves:
                    new_state = deepcopy(node.state)
                    new_state.make_move(move)
                    child_node = MCTSNode(state=new_state, parent=node, action=move)
                    node.children.append(child_node)
                    return child_node
        return None

    def _simplified_simulate(self, state, max_depth=10):
        """성능 향상된 시뮬레이션 (8% 스마트 + 92% 무작위)"""
        current_state = deepcopy(state)
        depth = 0
        
        while not current_state.is_terminal() and depth < max_depth:
            if current_state.current_piece is None:  # 말 선택 단계
                # 성능 우선: 8% 스마트 시뮬레이션
                if random.random() < 0.08:
                    piece = self._smart_piece_selection(current_state)
                else:
                    piece = random.choice(current_state.remaining_pieces)
                current_state.make_move(piece)
            else:  # 위치 선택 단계
                # 성능 우선: 8% 스마트 시뮬레이션
                if random.random() < 0.08:
                    move = self._smart_move_selection(current_state)
                else:
                    move = random.choice(current_state.get_legal_moves())
                current_state.make_move(move)
            depth += 1

        if current_state.is_terminal():
            return current_state.get_result()
        else:
            # 간소화된 휴리스틱 평가 (기존 4개 → 2개)
            return self._simple_heuristic_evaluation(current_state, state)

    def _smart_piece_selection(self, state):
        """스마트한 말 선택 (기존 로직 유지)"""
        safe_pieces = []
        risky_pieces = []
        
        for piece in state.remaining_pieces:
            if self._has_winning_move_fast(state, piece):  # fast 버전 사용
                risky_pieces.append(piece)
            else:
                safe_pieces.append(piece)
        
        if safe_pieces:
            return random.choice(safe_pieces)
        else:
            return random.choice(risky_pieces)

    def _smart_move_selection(self, state):
        """스마트한 위치 선택 (기존 로직 유지)"""
        legal_moves = state.get_legal_moves()
        
        # 1순위: 즉시 승리
        for move in legal_moves:
            try:
                temp_state = deepcopy(state)
                temp_state.make_move(move)
                if temp_state.check_victory():
                    return move
            except:
                continue
        
        # 2순위: 중앙
        center_positions = [(1,1), (1,2), (2,1), (2,2)]
        center_moves = [move for move in legal_moves if move in center_positions]
        if center_moves and random.random() < 0.7:
            return random.choice(center_moves)
        
        return random.choice(legal_moves)

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            if node.state.inturn == 0:
                if node.state.turn in [0,1]:
                    node.value += reward
                else:
                    node.value -= reward
            else:
                if node.state.turn in [2,3]:
                    node.value += reward
                else:
                    node.value -= reward
            node = node.parent

    def _simple_heuristic_evaluation(self, current_state, original_state):
        """간소화된 휴리스틱 평가 (기존 4개 → 2개)"""
        score = 0

        # 핵심 평가만 수행
        score -= self._evaluate_opponent_threat(current_state) * 1.0
        score += self._evaluate_piece_diversity(current_state) * 0.3
        
        # 플레이어 관점 조정
        if original_state.inturn == 0:
            return score if original_state.turn == 0 else -score
        elif original_state.inturn == 1:
            return score if original_state.turn == 2 else -score
        
        return score

    def _evaluate_opponent_threat(self, state):
        """기존 위협 평가 (로직 유지)"""
        lines = []

        for i in range(4):
            lines.append(state.board[i, :])
            lines.append(state.board[:, i])

        lines.append(np.diag(state.board))
        lines.append(np.diag(np.fliplr(state.board)))

        for i in range(3):
            for j in range(3):
                lines.append([
                    state.board[i, j], state.board[i, j + 1],
                    state.board[i + 1, j], state.board[i + 1, j + 1]
                ])

        score = 0
        for line in lines:
            empty_slots = sum(1 for cell in line if cell == 0)
            if empty_slots == 1:
                score -= 0.2

        for line in lines:
            if 0 in line:
                continue
            try:
                attributes = [state.all_pieces[int(cell) - 1] for cell in line if cell != 0]
                for i in range(4):
                    if all(attr[i] == attributes[0][i] for attr in attributes):
                        score += 0.2
            except:
                continue

        return score

    def _evaluate_piece_diversity(self, state):
        """기존 다양성 평가 (로직 유지)"""
        diversity_score = 0
        try:
            for piece in state.remaining_pieces:
                attributes = list(zip(*state.remaining_pieces))
                for i in range(4):
                    if attributes[i].count(piece[i]) == 1:
                        diversity_score -= 0.2
        except:
            pass
        return diversity_score


class QuartoState:
    def __init__(self, board=None, remaining_pieces=None, current_piece=None, turn=0, inturn=0):
        # 4x4 numpy 배열로 보드 초기화 (0으로 초기화)
        self.board = board if board is not None else np.zeros((4, 4), dtype=int)
        # 전체 말들 (0과 1로 이루어진 4개의 요소를 가진 튜플 배열)
        self.all_pieces = [(a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)]
        # 남은 말들 (초기에는 모든 말을 포함)
        self.remaining_pieces = remaining_pieces if remaining_pieces else self.all_pieces[:]
        # 현재 차례에 놓을 말
        self.current_piece = current_piece
        # 현재 턴 (0: 내가 말 선택, 1: 상대가 위치 선택, 2: 상대가 말 선택, 3: 내가 위치 선택)
        self.turn = turn
        # 마지막 움직임
        self.last_move = None
        # 마지막 말 선택
        self.last_piece = None

        self.inturn = inturn

    def get_legal_moves(self):
        # 비어있는 칸의 좌표 반환 (값이 0인 칸)
        return [(row, col) for row, col in zip(*np.where(self.board == 0))]

    def make_move(self, action):
        if self.turn in [0, 2]:  # 말 선택 단계
            if action not in self.remaining_pieces:
                raise ValueError(f"Invalid action: {action} is not available.")
            self.last_piece = action
            self.current_piece = action
        elif self.turn in [1, 3]:  # 위치 선택 단계
            if not isinstance(action, tuple) or len(action) != 2:
                raise ValueError(f"Invalid action: {action} must be a tuple (row, col).")
            row, col = action
            if self.board[row, col] != 0:
                raise ValueError(f"Illegal move: Position already occupied at {action}")
            piece_index = self.all_pieces.index(self.current_piece) + 1
            self.board[row, col] = piece_index
            self.remaining_pieces.remove(self.current_piece)
            self.last_move = action
            self.current_piece = None  # 다음 단계는 말 선택
        else:
            raise ValueError(f"Invalid turn: {self.turn}")

        # 턴 전환 (0 -> 1 -> 2 -> 3 -> 0)
        self.turn = (self.turn + 1) % 4

    def is_terminal(self):
        # 승리 조건 확인 또는 남은 말이 없는 경우 종료
        if self.check_victory():
            return True
        if not self.remaining_pieces and not self.get_legal_moves():
            return True
        return False

    def get_result(self):
        # 게임 결과 반환 (1: 승리, 0: 무승부, -1: 패배)
        if self.check_victory():
            if self.inturn == 0:
                return 1 if self.turn == 0 else -1
            elif self.inturn == 1:
                return 1 if self.turn == 2 else -1
        elif self.current_piece is None:
            return 0  # 모든 말을 사용했지만 승리 조건이 없음
        return -3

    def check_victory(self):
        # 가로, 세로, 대각선에서 승리 조건 확인
        for row in self.board:
            if self.is_quarto(row):
                return True
        for col in self.board.T:
            if self.is_quarto(col):
                return True
        if self.is_quarto(np.diag(self.board)) or self.is_quarto(np.diag(np.fliplr(self.board))):
            return True

        # 2x2 블록에서 승리 조건 확인
        for i in range(3):  # 0, 1, 2 (블록 시작 행)
            for j in range(3):  # 0, 1, 2 (블록 시작 열)
                block = [
                    self.board[i, j],
                    self.board[i, j + 1],
                    self.board[i + 1, j],
                    self.board[i + 1, j + 1]
                ]
                if self.is_quarto(block):
                    return True
        return False

    def is_quarto(self, line):
        # 4개의 말이 모두 채워져 있고, 공통 속성을 가진 경우
        if 0 in line:  # 값이 0인 경우 비어 있는 칸
            return False
        # 인덱스를 통해 all_pieces에서 속성을 가져옴
        try:
            properties = [self.all_pieces[int(piece) - 1] for piece in line]
            return any(all(prop[i] for prop in properties) or not any(prop[i] for prop in properties) for i in range(4))
        except:
            return False

# 기존 MCTS와 호환성을 위한 별칭
MCTS = MCTSPlus
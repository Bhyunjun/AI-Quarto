import numpy as np
import random
from itertools import product
import time
import copy
import math

class MCTSNode:
    def __init__(self, board, available_pieces, selected_piece, is_select_phase, move=None, parent=None):
        self.board = copy.deepcopy(board)
        self.available_pieces = list(available_pieces)
        self.selected_piece = selected_piece
        self.is_select_phase = is_select_phase
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = []

        if is_select_phase:
            self.untried_moves = list(available_pieces)
        else:
            self.untried_moves = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]

    def ucb1(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        return not any(0 in row for row in self.board) or len(self.available_pieces) == 0

class P1:
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = np.array(board) if isinstance(board, list) else board.copy()
        self.available_pieces = list(available_pieces)
        self.game_start_time = time.time()
        self.time_limit_per_game = 180.0  # 3분 = 180초
        self.simulations_per_piece = 100  # 각 조각당 시뮬레이션 횟수
        self.simulations_per_position = 150  # 각 위치당 시뮬레이션 횟수

    def get_remaining_time(self):
        """남은 시간 계산"""
        elapsed = time.time() - self.game_start_time
        return max(0, self.time_limit_per_game - elapsed)

    def select_piece(self, n_simulations=None):
        """각 조각을 상대가 받았을 때의 결과를 MCTS로 평가하여 최악의 조각 선택"""
        start_time = time.time()
        remaining_time = self.get_remaining_time()

        if remaining_time <= 0:
            return random.choice(self.available_pieces) if self.available_pieces else None

        if len(self.available_pieces) <= 1:
            return self.available_pieces[0] if self.available_pieces else None

        # 첫 수는 전략적 선택
        if len(self.available_pieces) == 16:
            # 극단적이지 않은 조각들을 선호
            balanced_pieces = [(0,1,0,1), (1,0,1,0), (0,0,1,1), (1,1,0,0)]
            for piece in balanced_pieces:
                if piece in self.available_pieces:
                    return piece
            return random.choice(self.available_pieces)

        # 각 조각별로 상대방이 얻을 수 있는 이점 평가
        piece_scores = {}
        time_per_piece = min(5.0, remaining_time / len(self.available_pieces))

        for piece in self.available_pieces:
            if time.time() - start_time > remaining_time:
                break

            # 이 조각으로 상대방이 얻을 수 있는 최대 이점 계산
            opponent_advantage = self.evaluate_piece_for_opponent_mcts(piece, time_per_piece)
            piece_scores[piece] = opponent_advantage

        # 상대방에게 가장 불리한 조각 선택 (점수가 낮을수록 좋음)
        if not piece_scores:
            return random.choice(self.available_pieces)

        best_piece = min(piece_scores.items(), key=lambda x: x[1])[0]
        return best_piece

    def evaluate_piece_for_opponent_mcts(self, piece, time_limit):
        """MCTS를 사용하여 특정 조각이 상대방에게 얼마나 유리한지 평가"""
        start_time = time.time()

        # 즉시 승리 가능한 조각인지 확인
        if self.is_immediately_dangerous(piece):
            return float('inf')  # 매우 위험한 조각

        # 각 가능한 위치에 대해 MCTS 실행
        empty_positions = [(r, c) for r in range(4) for c in range(4) if self.board[r, c] == 0]
        if not empty_positions:
            return 0.0

        total_score = 0.0
        positions_evaluated = 0

        # 각 위치에 대해 시뮬레이션
        for r, c in empty_positions:
            if time.time() - start_time > time_limit:
                break

            # 이 위치에 조각을 놓은 상태에서 MCTS 실행
            temp_board = self.board.copy()
            piece_idx = self.pieces.index(piece) + 1
            temp_board[r, c] = piece_idx

            # 상대방 관점에서 이 상태의 가치 평가 (위치 가중치 제외)
            position_score = self.evaluate_board_state_mcts_no_location(
                temp_board,
                [p for p in self.available_pieces if p != piece],
                False,  # 상대방 차례
                min(self.simulations_per_piece, int(time_limit * 20))
            )

            total_score += position_score
            positions_evaluated += 1

        if positions_evaluated == 0:
            return 0.0

        return total_score / positions_evaluated

    def place_piece(self, selected_piece, n_simulations=None):
        """각 가능한 위치에 대해 MCTS를 실행하여 최적의 위치 선택"""
        start_time = time.time()
        remaining_time = self.get_remaining_time()

        if remaining_time <= 0:
            empty_positions = [(r, c) for r in range(4) for c in range(4) if self.board[r, c] == 0]
            return random.choice(empty_positions) if empty_positions else (0, 0)

        empty_positions = [(r, c) for r in range(4) for c in range(4) if self.board[r, c] == 0]

        if not empty_positions:
            return (0, 0)
        if len(empty_positions) == 1:
            return empty_positions[0]

        # 즉시 승리 확인
        winning_move = self.find_winning_move(selected_piece)
        if winning_move:
            return winning_move

        # 각 위치별로 MCTS 평가
        position_scores = {}
        time_per_position = min(5.0, remaining_time / len(empty_positions))
        piece_idx = self.pieces.index(selected_piece) + 1

        for r, c in empty_positions:
            if time.time() - start_time > remaining_time:
                break

            # 이 위치에 놓았을 때의 상태 평가
            temp_board = self.board.copy()
            temp_board[r, c] = piece_idx

            # MCTS로 이 위치의 가치 평가 (내가 놓을 때는 위치 가중치 포함)
            position_value = self.evaluate_position_mcts(
                temp_board,
                self.available_pieces,
                r, c,
                time_per_position
            )

            position_scores[(r, c)] = position_value

        # 가장 높은 점수의 위치 선택
        if not position_scores:
            return self.select_position_heuristic(empty_positions, selected_piece)

        best_position = max(position_scores.items(), key=lambda x: x[1])[0]
        return best_position

    def evaluate_position_mcts(self, board, available_pieces, row, col, time_limit):
        """특정 위치의 가치를 MCTS로 평가"""
        start_time = time.time()

        # 즉시 승리인지 확인
        if self.check_win(board):
            return float('inf')

        # MCTS 루트 노드 생성 (상대방 차례로 시작)
        root = MCTSNode(board, available_pieces, None, True)

        simulations_done = 0
        max_simulations = self.simulations_per_position

        while simulations_done < max_simulations and (time.time() - start_time) < time_limit:
            # Selection
            node = self.select(root)

            # Simulation
            reward = self.simulate_from_position(node)

            # Backpropagation
            self.backpropagate(node, reward)

            simulations_done += 1

        # 이 위치의 평균 가치 반환
        if root.visits > 0:
            return root.value / root.visits
        else:
            # 기본 휴리스틱 평가
            return self.evaluate_position_heuristic(board, row, col)

    def evaluate_position_mcts_no_location(self, board, available_pieces, row, col, time_limit):
        """특정 위치의 가치를 MCTS로 평가 - 위치 가중치 제외"""
        start_time = time.time()

        # 즉시 승리인지 확인
        if self.check_win(board):
            return float('inf')

        # MCTS 루트 노드 생성 (상대방 차례로 시작)
        root = MCTSNode(board, available_pieces, None, True)

        simulations_done = 0
        max_simulations = self.simulations_per_position

        while simulations_done < max_simulations and (time.time() - start_time) < time_limit:
            # Selection
            node = self.select(root)

            # Simulation - 위치 가중치 제외된 시뮬레이션
            reward = self.simulate_from_position_no_location(node)

            # Backpropagation
            self.backpropagate(node, reward)

            simulations_done += 1

        # 이 위치의 평균 가치 반환
        if root.visits > 0:
            return root.value / root.visits
        else:
            # 기본 휴리스틱 평가 - 위치 가중치 제외
            return self.evaluate_position_heuristic_no_location(board, row, col)

    def evaluate_board_state_mcts(self, board, available_pieces, is_my_turn, max_simulations):
        """보드 상태를 MCTS로 평가"""
        root = MCTSNode(board, available_pieces, None, True)

        for _ in range(max_simulations):
            node = self.select(root)
            reward = self.simulate(node)

            # 턴에 따라 보상 조정
            if not is_my_turn:
                reward = -reward

            self.backpropagate(node, reward)

        if root.visits > 0:
            return root.value / root.visits
        else:
            return 0.0

    def evaluate_board_state_mcts_no_location(self, board, available_pieces, is_my_turn, max_simulations):
        """보드 상태를 MCTS로 평가 - 위치 가중치 제외"""
        root = MCTSNode(board, available_pieces, None, True)

        for _ in range(max_simulations):
            node = self.select(root)
            reward = self.simulate_no_location(node)

            # 턴에 따라 보상 조정
            if not is_my_turn:
                reward = -reward

            self.backpropagate(node, reward)

        if root.visits > 0:
            return root.value / root.visits
        else:
            return 0.0

    def simulate_from_position(self, node):
        """특정 노드에서 게임 끝까지 시뮬레이션"""
        sim_board = node.board.copy()
        sim_available = list(node.available_pieces)
        sim_selected = node.selected_piece
        sim_is_select = node.is_select_phase

        max_moves = 15
        moves_made = 0
        current_player_is_me = not node.is_select_phase  # 선택 페이즈면 상대 차례

        while moves_made < max_moves and sim_available and np.any(sim_board == 0):
            if sim_is_select:
                # 조각 선택
                if current_player_is_me:
                    # 내 차례: 상대에게 안 좋은 조각 선택
                    sim_selected = self.select_worst_piece_for_opponent(sim_available, sim_board)
                else:
                    # 상대 차례: 합리적인 선택
                    sim_selected = self.select_reasonable_piece(sim_available, sim_board)

                sim_available.remove(sim_selected)
                sim_is_select = False
            else:
                # 조각 배치
                empty = [(r, c) for r in range(4) for c in range(4) if sim_board[r, c] == 0]
                if empty:
                    piece_idx = self.pieces.index(sim_selected) + 1

                    # 즉시 승리 위치 확인
                    win_pos = self.find_winning_position(sim_board, piece_idx, empty)

                    if win_pos:
                        r, c = win_pos
                    else:
                        # 전략적 위치 선택
                        if current_player_is_me or random.random() < 0.7:
                            r, c = self.select_strategic_position(empty, sim_board, sim_selected)
                        else:
                            r, c = random.choice(empty)

                    sim_board[r, c] = piece_idx

                    if self.check_win(sim_board):
                        return 1.0 if current_player_is_me else -1.0

                    sim_is_select = True
                    current_player_is_me = not current_player_is_me
                else:
                    break

            moves_made += 1

        # 휴리스틱 평가
        evaluation = self.evaluate_position(sim_board)
        return evaluation if current_player_is_me else -evaluation

    def simulate_from_position_no_location(self, node):
        """특정 노드에서 게임 끝까지 시뮬레이션 - 위치 가중치 제외"""
        sim_board = node.board.copy()
        sim_available = list(node.available_pieces)
        sim_selected = node.selected_piece
        sim_is_select = node.is_select_phase

        max_moves = 15
        moves_made = 0
        current_player_is_me = not node.is_select_phase

        while moves_made < max_moves and sim_available and np.any(sim_board == 0):
            if sim_is_select:
                # 조각 선택
                if current_player_is_me:
                    # 내 차례: 상대에게 안 좋은 조각 선택
                    sim_selected = self.select_worst_piece_for_opponent(sim_available, sim_board)
                else:
                    # 상대 차례: 합리적인 선택
                    sim_selected = self.select_reasonable_piece(sim_available, sim_board)

                sim_available.remove(sim_selected)
                sim_is_select = False
            else:
                # 조각 배치
                empty = [(r, c) for r in range(4) for c in range(4) if sim_board[r, c] == 0]
                if empty:
                    piece_idx = self.pieces.index(sim_selected) + 1

                    # 즉시 승리 위치 확인
                    win_pos = self.find_winning_position(sim_board, piece_idx, empty)

                    if win_pos:
                        r, c = win_pos
                    else:
                        # 전략적 위치 선택 - 위치 가중치 제외
                        if current_player_is_me or random.random() < 0.7:
                            r, c = self.select_strategic_position_no_location(empty, sim_board, sim_selected)
                        else:
                            r, c = random.choice(empty)

                    sim_board[r, c] = piece_idx

                    if self.check_win(sim_board):
                        return 1.0 if current_player_is_me else -1.0

                    sim_is_select = True
                    current_player_is_me = not current_player_is_me
                else:
                    break

            moves_made += 1

        # 휴리스틱 평가 - 위치 가중치 제외
        evaluation = self.evaluate_position_no_location(sim_board)
        return evaluation if current_player_is_me else -evaluation

    def simulate(self, node):
        """기본 시뮬레이션"""
        return self.simulate_from_position(node)

    def simulate_no_location(self, node):
        """기본 시뮬레이션 - 위치 가중치 제외"""
        return self.simulate_from_position_no_location(node)

    def select_worst_piece_for_opponent(self, available_pieces, board):
        """상대에게 가장 안 좋은 조각 선택 (시뮬레이션용)"""
        if len(available_pieces) <= 3:
            # 조각이 적으면 빠른 평가
            worst_piece = None
            min_score = float('inf')

            for piece in available_pieces:
                score = self.quick_piece_evaluation(piece, board)
                if score < min_score:
                    min_score = score
                    worst_piece = piece

            return worst_piece if worst_piece else random.choice(available_pieces)
        else:
            # 조각이 많으면 샘플링
            sample = random.sample(available_pieces, min(3, len(available_pieces)))
            return min(sample, key=lambda p: self.quick_piece_evaluation(p, board))

    def select_reasonable_piece(self, available_pieces, board):
        """합리적인 조각 선택 (상대방 시뮬레이션용)"""
        # 즉시 위험하지 않은 조각 찾기
        safe_pieces = []
        for piece in available_pieces[:min(5, len(available_pieces))]:
            if not self.quick_danger_check(piece, board):
                safe_pieces.append(piece)

        return random.choice(safe_pieces) if safe_pieces else random.choice(available_pieces)

    def quick_piece_evaluation(self, piece, board):
        """조각의 빠른 평가 (낮을수록 상대에게 불리)"""
        piece_idx = self.pieces.index(piece) + 1
        score = 0.0

        # 즉시 승리 가능성
        for r in range(4):
            for c in range(4):
                if board[r, c] == 0:
                    board[r, c] = piece_idx
                    if self.check_win(board):
                        score += 10.0
                    else:
                        # 위협 생성 능력
                        threats = self.count_created_threats(board)
                        score += threats * 2.0
                    board[r, c] = 0

                    # 샘플링을 위해 몇 개만 확인
                    if score > 0:
                        break

        return score

    def find_winning_position(self, board, piece_idx, empty_positions):
        """즉시 승리 가능한 위치 찾기"""
        for r, c in empty_positions:
            board[r, c] = piece_idx
            if self.check_win(board):
                board[r, c] = 0
                return (r, c)
            board[r, c] = 0
        return None

    def select_strategic_position(self, empty_positions, board, piece):
        """전략적 위치 선택"""
        # 중앙 우선
        center = [(1, 1), (1, 2), (2, 1), (2, 2)]
        center_empty = [pos for pos in center if pos in empty_positions]

        if center_empty and random.random() < 0.7:
            return random.choice(center_empty)

        # 위협을 많이 만드는 위치
        best_positions = []
        max_threats = -1
        piece_idx = self.pieces.index(piece) + 1

        for r, c in empty_positions[:min(5, len(empty_positions))]:
            board[r, c] = piece_idx
            threats = self.count_created_threats(board)
            board[r, c] = 0

            if threats > max_threats:
                max_threats = threats
                best_positions = [(r, c)]
            elif threats == max_threats:
                best_positions.append((r, c))

        if best_positions:
            return random.choice(best_positions)

        return random.choice(empty_positions)

    def select_strategic_position_no_location(self, empty_positions, board, piece):
        """전략적 위치 선택 - 위치 가중치 제외"""
        # 위치 가중치 없이 순수하게 위협 기반으로만 선택
        best_positions = []
        max_threats = -1
        piece_idx = self.pieces.index(piece) + 1

        for r, c in empty_positions[:min(5, len(empty_positions))]:
            board[r, c] = piece_idx
            threats = self.count_created_threats(board)
            board[r, c] = 0

            if threats > max_threats:
                max_threats = threats
                best_positions = [(r, c)]
            elif threats == max_threats:
                best_positions.append((r, c))

        if best_positions:
            return random.choice(best_positions)

        return random.choice(empty_positions)

    def evaluate_position_heuristic(self, board, row, col):
        """위치의 휴리스틱 평가"""
        score = 0.0

        # 중앙 위치 보너스
        if (row, col) in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            score += 3.0
        elif (row, col) in [(0, 0), (0, 3), (3, 0), (3, 3)]:
            score += 2.0
        else:
            score += 1.0

        # 이 위치가 속한 라인들의 잠재력
        lines_through_position = self.get_lines_through_position(row, col)
        for line_coords in lines_through_position:
            line_values = [board[r, c] for r, c in line_coords]
            line_potential = self.evaluate_line_potential(line_values)
            score += line_potential * 0.5

        return score

    def evaluate_position_heuristic_no_location(self, board, row, col):
        """위치의 휴리스틱 평가 - 위치 가중치 제외"""
        score = 0.0

        # 위치 가중치 제거
        # 이 위치가 속한 라인들의 잠재력만 평가
        lines_through_position = self.get_lines_through_position(row, col)
        for line_coords in lines_through_position:
            line_values = [board[r, c] for r, c in line_coords]
            line_potential = self.evaluate_line_potential(line_values)
            score += line_potential * 0.5

        return score

    def get_lines_through_position(self, row, col):
        """특정 위치를 지나는 모든 라인"""
        lines = []

        # 행
        lines.append([(row, c) for c in range(4)])

        # 열
        lines.append([(r, col) for r in range(4)])

        # 주 대각선
        if row == col:
            lines.append([(i, i) for i in range(4)])

        # 부 대각선
        if row + col == 3:
            lines.append([(i, 3-i) for i in range(4)])

        return lines

    def evaluate_line_potential(self, line_values):
        """라인의 잠재력 평가"""
        empty_count = sum(1 for val in line_values if val == 0)

        if empty_count == 0 or empty_count == 4:
            return 0.0

        non_empty = [val for val in line_values if val != 0]
        pieces = [self.pieces[val-1] for val in non_empty]

        # 공통 특성이 있는지 확인
        for trait_idx in range(4):
            trait_values = [p[trait_idx] for p in pieces]
            if len(set(trait_values)) == 1:
                # 공통 특성이 있으면 빈 칸 수에 반비례하는 점수
                return (4 - empty_count) * 0.3

        return 0.0

    # 기존 MCTS 메서드들
    def select(self, node):
        """MCTS 선택 단계"""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = self.best_child(node)
        return node

    def expand(self, node):
        """MCTS 확장 단계"""
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)

        new_board = node.board.copy()
        new_available_pieces = list(node.available_pieces)

        if node.is_select_phase:
            new_selected_piece = move
            new_available_pieces.remove(move)
            new_is_select_phase = False
        else:
            piece_idx = self.pieces.index(node.selected_piece) + 1
            new_board[move[0]][move[1]] = piece_idx
            new_selected_piece = None
            new_is_select_phase = True

        child = MCTSNode(new_board, new_available_pieces, new_selected_piece,
                         new_is_select_phase, move, node)
        node.children.append(child)
        return child

    def best_child(self, node, c=1.414):
        """UCB1을 사용하여 최적의 자식 노드 선택"""
        return max(node.children, key=lambda child: child.ucb1(c))

    def backpropagate(self, node, reward):
        """MCTS 역전파 단계"""
        while node is not None:
            node.visits += 1
            node.value += reward
            reward = -reward  # 턴 기반 게임이므로 보상 반전
            node = node.parent

    # 유틸리티 메서드들
    def is_immediately_dangerous(self, piece):
        """조각이 즉시 위험한지 확인"""
        piece_idx = self.pieces.index(piece) + 1

        for r in range(4):
            for c in range(4):
                if self.board[r, c] == 0:
                    self.board[r, c] = piece_idx
                    if self.check_win(self.board):
                        self.board[r, c] = 0
                        return True
                    self.board[r, c] = 0

        return False

    def quick_danger_check(self, piece, board):
        """빠른 위험 체크"""
        piece_idx = self.pieces.index(piece) + 1

        # 최대 3개 위치만 확인
        checked = 0
        for r in range(4):
            for c in range(4):
                if board[r, c] == 0:
                    board[r, c] = piece_idx
                    if self.check_win(board):
                        board[r, c] = 0
                        return True
                    board[r, c] = 0

                    checked += 1
                    if checked >= 3:
                        return False

        return False

    def get_strategic_positions(self, empty_positions):
        """전략적으로 좋은 위치들 반환"""
        center = [(1, 1), (1, 2), (2, 1), (2, 2)]
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]

        strategic = []

        # 중앙 우선
        for pos in center:
            if pos in empty_positions:
                strategic.append(pos)

        # 모서리 다음
        for pos in corners:
            if pos in empty_positions:
                strategic.append(pos)

        # 나머지 위치
        if not strategic:
            strategic = empty_positions

        return strategic

    def evaluate_position(self, board):
        """보드 상태 평가"""
        score = 0.0

        # 승리에 가까운 정도
        win_proximity = self.evaluate_win_proximity(board)
        score += win_proximity * 0.5

        # 중앙 통제
        center_control = self.evaluate_center_control(board)
        score += center_control * 0.2

        # 위협 수
        threats = self.count_all_threats(board)
        score += threats * 0.3

        return max(-1.0, min(1.0, score))

    def evaluate_position_no_location(self, board):
        """보드 상태 평가 - 위치 가중치 제외"""
        score = 0.0

        # 승리에 가까운 정도
        win_proximity = self.evaluate_win_proximity(board)
        score += win_proximity * 0.5

        # 중앙 통제 제거
        # center_control = self.evaluate_center_control(board)
        # score += center_control * 0.2

        # 위협 수
        threats = self.count_all_threats(board)
        score += threats * 0.3

        return max(-1.0, min(1.0, score))

    def evaluate_win_proximity(self, board):
        """승리에 얼마나 가까운지 평가"""
        proximity_score = 0.0

        all_lines = self.get_all_winning_lines()

        for line_coords in all_lines:
            line_values = [board[r, c] for r, c in line_coords]
            empty_count = sum(1 for val in line_values if val == 0)

            if empty_count == 0:
                continue

            non_empty_values = [val for val in line_values if val != 0]
            if not non_empty_values:
                continue

            pieces = [self.pieces[val - 1] for val in non_empty_values]

            for trait_idx in range(4):
                trait_values = [p[trait_idx] for p in pieces]
                if len(set(trait_values)) == 1:
                    if empty_count == 1:
                        proximity_score += 0.8
                    elif empty_count == 2:
                        proximity_score += 0.3
                    break

        # 2x2 서브그리드
        for r in range(3):
            for c in range(3):
                subgrid = [board[r, c], board[r, c+1], board[r+1, c], board[r+1, c+1]]
                empty_count = sum(1 for val in subgrid if val == 0)

                if empty_count == 0 or empty_count == 4:
                    continue

                non_empty = [val for val in subgrid if val != 0]
                pieces = [self.pieces[val - 1] for val in non_empty]

                for trait_idx in range(4):
                    trait_values = [p[trait_idx] for p in pieces]
                    if len(set(trait_values)) == 1:
                        if empty_count == 1:
                            proximity_score += 0.6
                        elif empty_count == 2:
                            proximity_score += 0.2
                        break

        return proximity_score / 10.0

    def evaluate_center_control(self, board):
        """중앙 통제 평가"""
        center_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        control_score = 0.0

        for r, c in center_positions:
            if board[r, c] != 0:
                control_score += 0.25

        return control_score

    def count_all_threats(self, board):
        """모든 위협 계산"""
        threats = self.count_created_threats(board)

        # 2x2 위협 추가
        for r in range(3):
            for c in range(3):
                subgrid = [board[r, c], board[r, c+1], board[r+1, c], board[r+1, c+1]]
                empty_count = sum(1 for val in subgrid if val == 0)

                if empty_count == 1:
                    non_empty = [val for val in subgrid if val != 0]
                    if len(non_empty) == 3:
                        pieces = [self.pieces[val-1] for val in non_empty]
                        for trait_idx in range(4):
                            if len(set(p[trait_idx] for p in pieces)) == 1:
                                threats += 1
                                break

        return threats

    def get_all_winning_lines(self):
        """모든 승리 가능한 라인의 좌표 반환"""
        lines = []

        # 행
        for r in range(4):
            lines.append([(r, c) for c in range(4)])

        # 열
        for c in range(4):
            lines.append([(r, c) for r in range(4)])

        # 대각선
        lines.append([(i, i) for i in range(4)])
        lines.append([(i, 3-i) for i in range(4)])

        return lines

    def select_position_heuristic(self, empty_positions, piece):
        """휴리스틱을 사용한 위치 선택"""
        best_score = -float('inf')
        best_positions = []

        piece_idx = self.pieces.index(piece) + 1

        for r, c in empty_positions:
            score = 0.0

            # 위치의 기본 가치
            if (r, c) in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                score += 3
            elif (r, c) in [(0, 0), (0, 3), (3, 0), (3, 3)]:
                score += 2
            else:
                score += 1

            # 이 위치에 놓았을 때의 잠재력
            self.board[r, c] = piece_idx

            # 승리 가능성 확인
            if self.check_win(self.board):
                self.board[r, c] = 0
                return (r, c)

            # 위협 생성 능력
            threat_score = self.count_created_threats(self.board)
            score += threat_score * 2

            self.board[r, c] = 0

            if score > best_score:
                best_score = score
                best_positions = [(r, c)]
            elif score == best_score:
                best_positions.append((r, c))

        return random.choice(best_positions)

    def count_created_threats(self, board):
        """생성된 위협의 수 계산"""
        threats = 0

        all_lines = self.get_all_winning_lines()

        for line_coords in all_lines:
            line_values = [board[r, c] for r, c in line_coords]
            empty_count = sum(1 for val in line_values if val == 0)

            if empty_count == 1:
                non_empty = [val for val in line_values if val != 0]
                if len(non_empty) == 3:
                    pieces = [self.pieces[val - 1] for val in non_empty]
                    for trait_idx in range(4):
                        if len(set(p[trait_idx] for p in pieces)) == 1:
                            threats += 1
                            break

        return threats

    def find_winning_move(self, piece):
        """즉시 승리할 수 있는 수 찾기"""
        piece_idx = self.pieces.index(piece) + 1

        for r in range(4):
            for c in range(4):
                if self.board[r, c] == 0:
                    self.board[r, c] = piece_idx
                    if self.check_win(self.board):
                        self.board[r, c] = 0
                        return (r, c)
                    self.board[r, c] = 0

        return None

    def check_win(self, board):
        """승리 조건 확인"""
        # 행 확인
        for row in range(4):
            if self.check_line([board[row, col] for col in range(4)]):
                return True

        # 열 확인
        for col in range(4):
            if self.check_line([board[row, col] for row in range(4)]):
                return True

        # 대각선 확인
        if self.check_line([board[i, i] for i in range(4)]):
            return True
        if self.check_line([board[i, 3-i] for i in range(4)]):
            return True

        # 2x2 부분격자 확인
        for r in range(3):
            for c in range(3):
                subgrid = [board[r, c], board[r, c+1], board[r+1, c], board[r+1, c+1]]
                if 0 not in subgrid:
                    pieces_in_subgrid = [self.pieces[idx-1] for idx in subgrid]
                    for i in range(4):
                        if len(set(piece[i] for piece in pieces_in_subgrid)) == 1:
                            return True

        return False

    def check_line(self, line):
        """라인의 승리 조건 확인"""
        if 0 in line:
            return False

        pieces_in_line = [self.pieces[idx-1] for idx in line]

        for i in range(4):
            characteristic_values = [piece[i] for piece in pieces_in_line]
            if len(set(characteristic_values)) == 1:
                return True

        return False
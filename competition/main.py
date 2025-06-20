import numpy as np
import time
from multiprocessing import Pool, Manager
import copy
import traceback
import logging
from machines_p1 import P1
from machines_p2 import P2

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuartoGame:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.available_pieces = self.pieces[:]

    def check_line(self, line):
        if 0 in line:
            return False
        characteristics = np.array([self.pieces[piece_idx - 1] for piece_idx in line])
        for i in range(4):
            if len(set(characteristics[:, i])) == 1:
                return True
        return False

    def check_2x2_subgrid_win(self):
        for r in range(3):
            for c in range(3):
                subgrid = [self.board[r][c], self.board[r][c+1], self.board[r+1][c], self.board[r+1][c+1]]
                if 0 not in subgrid:
                    characteristics = [self.pieces[idx - 1] for idx in subgrid]
                    for i in range(4):
                        if len(set(char[i] for char in characteristics)) == 1:
                            return True
        return False

    def check_win(self):
        # Check rows
        for row in range(4):
            if self.check_line([self.board[row][col] for col in range(4)]):
                return True

        # Check columns
        for col in range(4):
            if self.check_line([self.board[row][col] for row in range(4)]):
                return True

        # Check diagonals
        if self.check_line([self.board[i][i] for i in range(4)]) or \
                self.check_line([self.board[i][3-i] for i in range(4)]):
            return True

        # Check 2x2 sub-grids
        if self.check_2x2_subgrid_win():
            return True

        return False

    def is_board_full(self):
        return not any(0 in row for row in self.board)

    def validate_piece_selection(self, selected_piece, selecting_player):
        """조각 선택 유효성 검사"""
        if selected_piece is None:
            return False, "선택된 조각이 None입니다"

        if selected_piece not in self.available_pieces:
            return False, f"선택된 조각 {selected_piece}이 사용 가능한 조각 목록에 없습니다"

        if not isinstance(selected_piece, tuple) or len(selected_piece) != 4:
            return False, f"잘못된 조각 형식: {selected_piece}"

        return True, ""

    def validate_piece_placement(self, board_row, board_col, placing_player):
        """조각 배치 유효성 검사"""
        if not isinstance(board_row, int) or not isinstance(board_col, int):
            return False, f"좌표가 정수가 아닙니다: ({board_row}, {board_col})"

        if not (0 <= board_row < 4 and 0 <= board_col < 4):
            return False, f"좌표가 범위를 벗어났습니다: ({board_row}, {board_col})"

        if self.board[board_row][board_col] != 0:
            return False, f"이미 조각이 있는 위치입니다: ({board_row}, {board_col})"

        return True, ""

    def safe_player_action(self, player_class, action_type, game_id, player_num, **kwargs):
        """플레이어 행동을 안전하게 실행"""
        try:
            if player_num == 1:
                player = P1(board=copy.deepcopy(self.board),
                            available_pieces=list(self.available_pieces))
            else:
                player = P2(board=copy.deepcopy(self.board),
                            available_pieces=list(self.available_pieces))

            if action_type == "select_piece":
                result = player.select_piece()
                logger.info(f"게임 {game_id}: P{player_num}가 조각 {result}를 선택했습니다")
                return result, None
            elif action_type == "place_piece":
                result = player.place_piece(kwargs['selected_piece'])
                logger.info(f"게임 {game_id}: P{player_num}가 조각을 ({result[0]}, {result[1]})에 배치했습니다")
                return result, None

        except Exception as e:
            error_msg = f"P{player_num} {action_type} 오류: {str(e)}"
            logger.error(f"게임 {game_id}: {error_msg}")
            logger.error(f"상세 오류:\n{traceback.format_exc()}")
            return None, error_msg

    def play_game(self, game_id):
        """단일 게임 실행"""
        logger.info(f"게임 {game_id} 시작")

        turn = 1
        flag = "select_piece"
        selected_piece = None
        time_consumption = {1: 0, 2: 0}
        move_count = 0
        errors = []

        while True:
            move_count += 1

            if flag == "select_piece":
                # 선택 플레이어
                selecting_player = 3 - turn

                start_time = time.time()
                result, error = self.safe_player_action(
                    P1 if selecting_player == 1 else P2,
                    "select_piece",
                    game_id,
                    selecting_player
                )
                end_time = time.time()

                time_consumption[selecting_player] += (end_time - start_time)

                if error:
                    errors.append(f"무브 {move_count}: {error}")
                    logger.error(f"게임 {game_id}: 조각 선택 실패 - {error}")
                    return {
                        'game_id': game_id,
                        'winner': 3 - selecting_player,  # 상대방 승리
                        'moves': move_count,
                        'time_p1': time_consumption[1],
                        'time_p2': time_consumption[2],
                        'final_board': self.board.copy(),
                        'error': f'P{selecting_player} 조각 선택 오류',
                        'error_details': error,
                        'all_errors': errors
                    }

                # 조각 선택 유효성 검사
                is_valid, validation_error = self.validate_piece_selection(result, selecting_player)
                if not is_valid:
                    error_msg = f"P{selecting_player} 잘못된 조각 선택: {validation_error}"
                    errors.append(f"무브 {move_count}: {error_msg}")
                    logger.error(f"게임 {game_id}: {error_msg}")
                    return {
                        'game_id': game_id,
                        'winner': 3 - selecting_player,
                        'moves': move_count,
                        'time_p1': time_consumption[1],
                        'time_p2': time_consumption[2],
                        'final_board': self.board.copy(),
                        'error': f'P{selecting_player} 잘못된 조각 선택',
                        'error_details': validation_error,
                        'all_errors': errors
                    }

                selected_piece = result
                flag = "place_piece"

            elif flag == "place_piece":
                # 배치 플레이어
                start_time = time.time()
                result, error = self.safe_player_action(
                    P1 if turn == 1 else P2,
                    "place_piece",
                    game_id,
                    turn,
                    selected_piece=selected_piece
                )
                end_time = time.time()

                time_consumption[turn] += (end_time - start_time)

                if error:
                    errors.append(f"무브 {move_count}: {error}")
                    logger.error(f"게임 {game_id}: 조각 배치 실패 - {error}")
                    return {
                        'game_id': game_id,
                        'winner': 3 - turn,  # 상대방 승리
                        'moves': move_count,
                        'time_p1': time_consumption[1],
                        'time_p2': time_consumption[2],
                        'final_board': self.board.copy(),
                        'error': f'P{turn} 조각 배치 오류',
                        'error_details': error,
                        'all_errors': errors
                    }

                board_row, board_col = result

                # 조각 배치 유효성 검사
                is_valid, validation_error = self.validate_piece_placement(board_row, board_col, turn)
                if not is_valid:
                    error_msg = f"P{turn} 잘못된 조각 배치: {validation_error}"
                    errors.append(f"무브 {move_count}: {error_msg}")
                    logger.error(f"게임 {game_id}: {error_msg}")
                    return {
                        'game_id': game_id,
                        'winner': 3 - turn,
                        'moves': move_count,
                        'time_p1': time_consumption[1],
                        'time_p2': time_consumption[2],
                        'final_board': self.board.copy(),
                        'error': f'P{turn} 잘못된 조각 배치',
                        'error_details': validation_error,
                        'all_errors': errors
                    }

                # 조각 배치 실행
                self.board[board_row][board_col] = self.pieces.index(selected_piece) + 1
                self.available_pieces.remove(selected_piece)
                selected_piece = None

                # 승리 확인
                if self.check_win():
                    logger.info(f"게임 {game_id}: P{turn} 승리!")
                    return {
                        'game_id': game_id,
                        'winner': turn,
                        'moves': move_count,
                        'time_p1': time_consumption[1],
                        'time_p2': time_consumption[2],
                        'final_board': self.board.copy(),
                        'all_errors': errors if errors else None
                    }
                elif self.is_board_full():
                    logger.info(f"게임 {game_id}: 무승부!")
                    return {
                        'game_id': game_id,
                        'winner': 0,  # Draw
                        'moves': move_count,
                        'time_p1': time_consumption[1],
                        'time_p2': time_consumption[2],
                        'final_board': self.board.copy(),
                        'all_errors': errors if errors else None
                    }
                else:
                    turn = 3 - turn
                    flag = "select_piece"

def play_single_game(game_id):
    """프로세스에서 실행될 단일 게임"""
    try:
        game = QuartoGame()
        result = game.play_game(game_id)
        return result
    except Exception as e:
        logger.error(f"게임 {game_id} 실행 중 예상치 못한 오류: {str(e)}")
        logger.error(f"상세 오류:\n{traceback.format_exc()}")
        return {
            'game_id': game_id,
            'winner': -1,  # 시스템 오류
            'moves': 0,
            'time_p1': 0,
            'time_p2': 0,
            'final_board': np.zeros((4, 4)),
            'error': '시스템 오류',
            'error_details': str(e)
        }

def main():
    print("=== Quarto 자동 대전 시스템 ===")
    print("P1 vs P2 - 20판 대전\n")

    # 멀티프로세싱을 사용한 병렬 실행
    num_games = 20
    num_processes = 4  # CPU 코어 수에 맞게 조정

    with Pool(processes=num_processes) as pool:
        # 게임 실행
        results = pool.map(play_single_game, range(1, num_games + 1))

    # 결과 분석
    p1_wins = sum(1 for r in results if r['winner'] == 1)
    p2_wins = sum(1 for r in results if r['winner'] == 2)
    draws = sum(1 for r in results if r['winner'] == 0)
    errors = sum(1 for r in results if 'error' in r and r['winner'] not in [1, 2, 0])

    total_p1_time = sum(r['time_p1'] for r in results)
    total_p2_time = sum(r['time_p2'] for r in results)

    valid_games = [r for r in results if r['winner'] in [1, 2, 0]]
    avg_moves = sum(r['moves'] for r in valid_games) / len(valid_games) if valid_games else 0

    # 오류 게임 분석
    error_games = [r for r in results if 'error' in r]
    p1_error_games = [r for r in error_games if 'P1' in r.get('error', '')]
    p2_error_games = [r for r in error_games if 'P2' in r.get('error', '')]

    # 결과 출력
    print("\n=== 대전 결과 ===")
    print(f"총 게임 수: {num_games}")
    print(f"P1 승리: {p1_wins}판 ({p1_wins/num_games*100:.1f}%)")
    print(f"P2 승리: {p2_wins}판 ({p2_wins/num_games*100:.1f}%)")
    print(f"무승부: {draws}판 ({draws/num_games*100:.1f}%)")
    if errors > 0:
        print(f"오류로 인한 게임 종료: {errors}판 ({errors/num_games*100:.1f}%)")
        print(f"  - P1 오류: {len(p1_error_games)}판")
        print(f"  - P2 오류: {len(p2_error_games)}판")

    if valid_games:
        print(f"\n평균 수: {avg_moves:.1f}")
    print(f"\n총 소요 시간:")
    print(f"  P1: {total_p1_time:.2f}초 (평균: {total_p1_time/num_games:.2f}초/게임)")
    print(f"  P2: {total_p2_time:.2f}초 (평균: {total_p2_time/num_games:.2f}초/게임)")

    # 오류 상세 정보
    if error_games:
        print("\n=== 오류 게임 상세 정보 ===")
        for r in error_games:
            print(f"게임 {r['game_id']}: {r.get('error', '알 수 없는 오류')}")
            if 'error_details' in r:
                print(f"  상세: {r['error_details']}")
            if 'all_errors' in r and r['all_errors']:
                print(f"  전체 오류 목록:")
                for err in r['all_errors']:
                    print(f"    - {err}")

    # 각 게임 상세 결과
    print("\n=== 게임별 상세 결과 ===")
    print("게임 | 승자     | 수 | P1 시간(초) | P2 시간(초) | 오류")
    print("-" * 70)
    for r in results:
        if r['winner'] > 0:
            winner_str = f"P{r['winner']}"
        elif r['winner'] == 0:
            winner_str = "무승부"
        else:
            winner_str = "시스템오류"

        error_str = r.get('error', '')[:15] if 'error' in r else ""
        print(f"{r['game_id']:4d} | {winner_str:8s} | {r['moves']:2d} | {r['time_p1']:10.3f} | {r['time_p2']:10.3f} | {error_str}")

    # 승률 그래프 (간단한 텍스트 버전)
    print("\n=== 승률 시각화 ===")
    p1_bar = "█" * int(p1_wins / num_games * 50)
    p2_bar = "█" * int(p2_wins / num_games * 50)
    draw_bar = "█" * int(draws / num_games * 50)

    if errors > 0:
        error_bar = "█" * int(errors / num_games * 50)
        print(f"P1: {p1_bar} {p1_wins/num_games*100:.1f}%")
        print(f"P2: {p2_bar} {p2_wins/num_games*100:.1f}%")
        print(f"무: {draw_bar} {draws/num_games*100:.1f}%")
        print(f"오류: {error_bar} {errors/num_games*100:.1f}%")
    else:
        print(f"P1: {p1_bar} {p1_wins/num_games*100:.1f}%")
        print(f"P2: {p2_bar} {p2_wins/num_games*100:.1f}%")
        print(f"무: {draw_bar} {draws/num_games*100:.1f}%")

if __name__ == "__main__":
    main()
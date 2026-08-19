"""
胜因分析 — 判定自对弈中 P1/P2 获胜的具体方式

纯规则盘面分析, 不依赖神经网络, 与 mcts.py 战术层语义保持一致:
  - 四线威胁/活四: 落子点所在直线方向存在空位, 补一手即成五连 (含冲四/跳四);
  - 活三威胁:       落子点所在方向存在空位, 补一手即成"四连且两端皆空" (含跳活三);
  - 一子双杀:       取整局中获胜方第一次形成双威胁/活四的那一手
                    (之后对手无法一手化解, 因此视为"制胜原因");
  - VCF/VCT:        扫描获胜方倒数连续的冲四/先手序列 (近似判定,
                    不校验对手应手是否被迫)。

输出分类 (与用户表格一致, 另补充普通五连兜底):
  类别              | 名称        | 说明
  直接必胜棋型      | 活四        | 一手形成两端皆空的四连, 对手无法一手防守
                    | 长连        | 六子或以上连成一线 (无禁手下直接判胜)
  一子双杀(经典)    | 双活三(三三)| 一手同时形成两个活三
                    | 双冲四(四四)| 一手同时形成两个冲四/四线威胁
                    | 活三冲四(四三)| 一手同时形成一个活三和一个冲四
  连续进攻战术      | VCF         | 连续冲四, 逼对手步步只能堵唯一位置直至连五
                    | VCT         | 连续活三/冲四等先手手段直至取胜
  普通取胜(兜底)    | 连五        | 普通五连 (冲四成五 / 跳四成五 等)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from mcts import _check_win_static, _run_extents


DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


@dataclass
class WinReason:
    """一局的胜因"""
    name: str                       # 中文名: 双活三 / 双冲四 / 活三冲四 / 活四 / 长连 / VCF / VCT / 连五
    category: str                   # 直接必胜棋型 / 一子双杀(经典) / 连续进攻战术 / 普通取胜
    detail: str = ''                # 补充说明
    move_no: int = 0                # 决定性一手 (1-based 全盘手数, 0=未知)
    move: Optional[Tuple[int, int]] = None   # 决定性一手坐标 (x, y)
    line_len: int = 0               # 终局连子长度


@dataclass
class WinGameSummary:
    """紧凑终局摘要, 用于跨进程传输并由 train_alphazero.py 交给分类器"""
    game_id: int
    winner: int                     # 1 / 2 / 0(平局)
    main_player: int
    board: np.ndarray               # 终局棋盘 int32 (H, W)
    moves: np.ndarray               # (N, 3) int32: [player, x, y], 含随机开局
    opening_count: int              # 前 opening_count 手为随机开局 (不计入战术判定)
    win_move: Optional[Tuple[int, int]] = None


# ============================================================
# 棋形辅助
# ============================================================

def _line_cells(board: np.ndarray, x: int, y: int, dx: int, dy: int,
                player: int, win_condition: int) -> Tuple[List[int], int]:
    """
    落子点 (x,y) 所在方向线段 (已含该子), 越界记为 -1 (视为阻挡)。
    线段长度 = 2*(win_condition-1)+1, 中心即落子点。
    返回 (cells, center_index)。
    """
    h, w = board.shape
    half = win_condition - 1
    cells = []
    for k in range(-half, win_condition):
        nx, ny = x + k * dx, y + k * dy
        if 0 <= nx < h and 0 <= ny < w:
            cells.append(int(board[nx, ny]))
        else:
            cells.append(-1)
    return cells, half


def _direction_five_cells(board: np.ndarray, x: int, y: int, dx: int, dy: int,
                          player: int, win_condition: int) -> int:
    """
    落子 (x,y) 后, 该方向线段内"补一手即可连五"的空位数:
      0 = 无四线威胁; 1 = 冲四/跳四; >=2 = 活四 (两个胜点)。
    """
    cells, _ = _line_cells(board, x, y, dx, dy, player, win_condition)
    n = len(cells)
    five = 0
    for e in range(n):
        if cells[e] != 0:
            continue
        cells[e] = player
        for j in range(n - win_condition + 1):
            if all(v == player for v in cells[j:j + win_condition]):
                five += 1
                break
        cells[e] = 0
    return five


def _direction_live_three(board: np.ndarray, x: int, y: int, dx: int, dy: int,
                          player: int, win_condition: int) -> bool:
    """
    落子 (x,y) 后, 该方向是否存在活三线:
    存在一个空位, 补一手后形成 "win_condition-1 连且两端皆空" (含跳活三)。
    与 mcts._find_double_threat_cells 的活三判定一致。
    """
    cells, _ = _line_cells(board, x, y, dx, dy, player, win_condition)
    n = len(cells)
    for e in range(n):
        if cells[e] != 0:
            continue
        cells[e] = player
        left, right = _run_extents(cells, e, player)
        cells[e] = 0
        run = left + right + 1
        if run == win_condition - 1:
            le = e - left - 1
            re = e + right + 1
            if (0 <= le < n and cells[le] == 0 and
                    0 <= re < n and cells[re] == 0):
                return True
    return False


def _move_threat_profile(board: np.ndarray, x: int, y: int, player: int,
                         win_condition: int
                         ) -> Tuple[int, int, int]:
    """
    落子后的威胁画像: (四线方向数, 活三方向数, 单方向最大胜点数)。
    单方向最大胜点数 >= 2 表示该方向是活四。
    """
    four_dirs = 0
    live3_dirs = 0
    max_four_cells = 0
    for dx, dy in DIRECTIONS:
        fc = _direction_five_cells(board, x, y, dx, dy, player, win_condition)
        if fc:
            four_dirs += 1
            max_four_cells = max(max_four_cells, fc)
        elif _direction_live_three(board, x, y, dx, dy, player, win_condition):
            live3_dirs += 1
    return four_dirs, live3_dirs, max_four_cells


def _classify_decisive(move_no: int, x: int, y: int,
                       four_dirs: int, live3_dirs: int, max_four_cells: int
                       ) -> Optional[WinReason]:
    """一手形成 活四 / 双冲四 / 活三冲四 / 双活三 时返回胜因, 否则 None"""
    if max_four_cells >= 2:
        return WinReason(
            '活四', '直接必胜棋型',
            '一手形成两端皆空的活四, 对手无法一手防守',
            move_no=move_no, move=(x, y))
    if four_dirs >= 2:
        return WinReason(
            '双冲四', '一子双杀(经典)',
            '一手形成双四(双冲四), 对手只能挡一个',
            move_no=move_no, move=(x, y))
    if four_dirs == 1 and live3_dirs >= 1:
        return WinReason(
            '活三冲四', '一子双杀(经典)',
            '一手同时形成活三与冲四',
            move_no=move_no, move=(x, y))
    if live3_dirs >= 2:
        return WinReason(
            '双活三', '一子双杀(经典)',
            '一手形成两个活三(三三), 对手只能挡一个',
            move_no=move_no, move=(x, y))
    return None


def _classify_terminal(final_board: np.ndarray, pre_board: np.ndarray,
                       tx: int, ty: int, player: int,
                       win_condition: int, move_no: int) -> WinReason:
    """终局连五/长连分类 (在未发现更早制胜棋形时使用)"""
    h, w = final_board.shape
    max_run = 0
    win_dir = None
    for dx, dy in DIRECTIONS:
        # 用棋盘坐标直接数连续子, 避免固定窗口截断超长连
        left = right = 0
        nx, ny = tx - dx, ty - dy
        while 0 <= nx < h and 0 <= ny < w and final_board[nx, ny] == player:
            left += 1
            nx -= dx
            ny -= dy
        nx, ny = tx + dx, ty + dy
        while 0 <= nx < h and 0 <= ny < w and final_board[nx, ny] == player:
            right += 1
            nx += dx
            ny += dy
        run = left + right + 1
        if run > max_run:
            max_run = run
            win_dir = (dx, dy)

    if max_run >= win_condition + 1:
        return WinReason(
            '长连', '直接必胜棋型', f'{max_run}子长连',
            move_no=move_no, move=(tx, ty), line_len=max_run)
    if max_run < win_condition:
        return WinReason(
            '连五', '普通取胜', '终局五连(无法解析棋形)',
            move_no=move_no, move=(tx, ty), line_len=max_run)

    # 找包含胜手的 wc 连续窗口, 计算胜手在五连内的偏移 k (0..wc-1)
    dx, dy = win_dir
    cells, half = _line_cells(final_board, tx, ty, dx, dy,
                              player, win_condition)
    n = len(cells)
    k = None
    for start in range(n - win_condition + 1):
        if all(v == player for v in cells[start:start + win_condition]):
            k = half - start
            break
    if k is None:
        return WinReason(
            '连五', '普通取胜', '终局五连(无法解析棋形)',
            move_no=move_no, move=(tx, ty), line_len=win_condition)

    # 五连两端棋盘坐标 (偏移 0 与 wc-1)
    x1, y1 = tx - k * dx, ty - k * dy
    x2, y2 = tx + (win_condition - 1 - k) * dx, ty + (win_condition - 1 - k) * dy

    if k == 0 or k == win_condition - 1:
        # 胜手在五连端部 → 终局前是连续四连, 看另一侧是否开放 (活四)
        if k == 0:
            ex, ey = x2 + dx, y2 + dy
        else:
            ex, ey = x1 - dx, y1 - dy
        open_end = (0 <= ex < h and 0 <= ey < w and pre_board[ex, ey] == 0)
        if open_end:
            return WinReason(
                '活四', '直接必胜棋型', '终局前已是两端开放的四连',
                move_no=move_no, move=(tx, ty), line_len=win_condition)
        return WinReason(
            '连五', '普通取胜', '冲四成五',
            move_no=move_no, move=(tx, ty), line_len=win_condition)
    return WinReason(
        '连五', '普通取胜', '跳四成五(断点补五)',
        move_no=move_no, move=(tx, ty), line_len=win_condition)


# ============================================================
# 终局重建 & 胜因判定 (公开 API)
# ============================================================

def reconstruct_game(game_result, win_condition: int = 5
                     ) -> Optional[WinGameSummary]:
    """
    从 GameResult 重建终局棋盘与完整手顺, 供胜因判定。
    平局 (winner==0) 或无步数时返回 None。

    重建原理: 每步 state 记录的是"落子前"的棋盘, 相邻 state 差出一个新子,
    归属上一步的 current_player; 终局胜手不在任何 state 中, 通过在最后一步
    状态中"补一手即连五"的唯一空位确定。
    """
    steps = game_result.steps
    winner = game_result.winner
    if winner == 0 or not steps:
        return None

    h, w = steps[0].state.shape[1], steps[0].state.shape[2]

    def board_from_state(state: np.ndarray,
                         current_player: int) -> np.ndarray:
        # 通道 0 = 当前玩家棋子, 通道 1 = 对手棋子 (见 self_play._build_state)
        b = np.zeros((h, w), dtype=np.int32)
        b[state[0] > 0.5] = current_player
        b[state[1] > 0.5] = 3 - current_player
        return b

    boards = [board_from_state(s.state, s.player) for s in steps]

    # 随机开局 = 第一步状态里已有的棋子 (落子先后对战术判定无影响)
    opening = []
    for p in (1, 2):
        for x, y in np.argwhere(boards[0] == p).tolist():
            opening.append((p, int(x), int(y)))
    opening_count = len(opening)

    # 中间手: 相邻状态差出一个新棋子, 归属上一步的 current_player
    moves = []
    prev = boards[0]
    for i in range(len(steps) - 1):
        diff = boards[i + 1] != prev
        idxs = np.argwhere(diff)
        if len(idxs) == 1:
            x, y = int(idxs[0][0]), int(idxs[0][1])
            moves.append((steps[i].player, x, y))
        prev = boards[i + 1]

    # 终局胜手: 最后一步状态中补一手即连五的唯一空位
    before = boards[-1].copy()
    win_move = None
    for x in range(h):
        for y in range(w):
            if before[x, y] != 0:
                continue
            before[x, y] = winner
            if _check_win_static(before, x, y, win_condition):
                win_move = (x, y)
                break
            before[x, y] = 0
        if win_move is not None:
            break

    final = before.copy()
    if win_move is not None:
        moves.append((winner, win_move[0], win_move[1]))
        final[win_move] = winner

    return WinGameSummary(
        game_id=game_result.game_id,
        winner=winner,
        main_player=game_result.main_player,
        board=final,
        moves=np.array(opening + moves, dtype=np.int32),
        opening_count=opening_count,
        win_move=win_move,
    )


def classify_game(summary, win_condition: int = 5) -> Optional[WinReason]:
    """
    判定一局 (非平局) 的胜因。
    返回 WinReason 或 None (平局/无法解析)。

    判定优先级:
      1. 整局中获胜方第一次形成的 活四/双冲四/活三冲四/双活三 (制胜棋形);
      2. 获胜方倒数连续的冲四序列 (VCF) / 活三冲四等先手序列 (VCT);
      3. 终局连五/长连分类。
    """
    if summary is None or summary.winner == 0:
        return None
    winner = int(summary.winner)
    moves = summary.moves
    opening_count = int(summary.opening_count)

    board = np.zeros_like(summary.board)
    for p, x, y in moves[:opening_count]:
        board[int(x), int(y)] = int(p)

    playable = moves[opening_count:]
    n = len(playable)
    if n == 0:
        return WinReason('连五', '普通取胜', '无有效手顺',
                         move=summary.win_move)

    winner_moves = []       # (move_no, x, y, four_dirs, live3_dirs, max_four_cells)
    pre_terminal = None
    for idx, (p, x, y) in enumerate(playable):
        p, x, y = int(p), int(x), int(y)
        move_no = opening_count + idx + 1
        if idx == n - 1:
            pre_terminal = board.copy()     # 终局落子前状态
            board[x, y] = p
            continue
        board[x, y] = p
        if p != winner:
            continue
        four_dirs, live3_dirs, max_four_cells = _move_threat_profile(
            board, x, y, winner, win_condition)
        reason = _classify_decisive(move_no, x, y, four_dirs,
                                    live3_dirs, max_four_cells)
        if reason is not None:
            return reason
        winner_moves.append((move_no, x, y, four_dirs,
                             live3_dirs, max_four_cells))

    if pre_terminal is None:
        return WinReason('连五', '普通取胜', '无法解析终局',
                         move=summary.win_move)

    terminal = board.copy()
    tp, tx, ty = int(playable[-1][0]), int(playable[-1][1]), int(playable[-1][2])

    # VCF / VCT: 从终局前一手倒推获胜方连续的先手序列
    vcf = vct = 0
    vcf_start = vct_start = 0
    for move_no, x, y, fd, l3, mfc in reversed(winner_moves):
        pure_four = (fd == 1 and l3 == 0 and mfc == 1)
        any_threat = (fd >= 1 or l3 >= 1)
        if pure_four:
            vcf += 1
            vct += 1
            vcf_start = vct_start = move_no
        elif any_threat:
            vcf = 0
            vct += 1
            vct_start = move_no
        else:
            break
    if vcf >= 2:
        return WinReason(
            'VCF', '连续进攻战术', f'连续{vcf}手冲四逼防后连五',
            move_no=vcf_start, move=(tx, ty), line_len=win_condition)
    if vct >= 2:
        return WinReason(
            'VCT', '连续进攻战术', f'连续{vct}手活三/冲四先手取胜',
            move_no=vct_start, move=(tx, ty), line_len=win_condition)

    return _classify_terminal(terminal, pre_terminal, tx, ty, winner,
                              win_condition, opening_count + n)


def print_win_reasons(games, win_condition: int = 5):
    """
    打印一轮对局中所有非平局局面的胜因。

    Args:
        games: SelfPlayManager.last_games, 形如
               [(winner, main_player, summary_or_None), ...]
        win_condition: 连子数 (默认 5)

    Returns:
        Counter: {胜因名称: 次数}
    """
    from collections import Counter
    counts = Counter()
    n_wins = 0
    for winner, main_player, summary in (games or []):
        if winner == 0 or summary is None:
            continue
        reason = classify_game(summary, win_condition)
        if reason is None:
            continue
        counts[reason.name] += 1
        n_wins += 1
        pos = f"第{reason.move_no}手" if reason.move_no > 0 else ""
        mv = (f"({reason.move[0]},{reason.move[1]})"
              if reason.move is not None else "")
        suffix = f" | {reason.detail}" if reason.detail else ""
        line = (f"    [胜因] Game {summary.game_id}: P{winner} 获胜 — "
                f"{reason.name}({reason.category})")
        if pos:
            line += f" | {pos} {mv}"
        line += suffix
        print(line)
    if n_wins:
        dist = ', '.join(f"{name}×{c}" for name, c in counts.most_common())
        print(f"    [胜因汇总] {n_wins} 局分胜负: {dist}")
    return counts

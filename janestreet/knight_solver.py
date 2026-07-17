"""
Knight-score puzzle — RELAXED (less-constrained) pass.

We ignore the board geometry and the knight's-L movement entirely.
Goal: find a sequence of operations (one per move N = 1, 2, 3, ...) that
reproduces the recorded score checkpoints.

Rules per move N:
    same : score += N
    up   : score *= N
    down : score /= N          (only if score % N == 0)

Altitude:
    The knight may only ever be ONE level above its base. So it can go
    "up" at most once before it must come back "down". We model altitude
    as a level in {0, 1}:
        up   : requires level == 0, sets level -> 1   (multiply)
        down : requires level == 1, sets level -> 0   (divide)
        same : level unchanged                        (add N)
    (This is the "you can only go up one level, and down once the score
    is divisible by N" reading.)

Checkpoints (score written down on arrival):
    moves 3, 6, 9, 12, 15, 18, then every K moves for some K > 3.
    Recorded values, in order (0 is the starting score before move 1):
        0 -> 37, 1100, 23, 138, 528, 449, 16, 750, 88, 272, 1
"""

import argparse
from itertools import product

from z3 import *

# ---- puzzle data -----------------------------------------------------------

# Recorded scores in order. First (0) is the start, before any move.
RECORDED = [0, 37, 1100, 23, 138, 528, 449, 16, 750, 88, 272, 1]

# Early checkpoints are fixed at every 3rd move up to 18.
EARLY_CHECKPOINTS = [3, 6, 9, 12, 15, 18]   # 6 checkpoints -> RECORDED[1..6]

# After move 18 the knight records every K moves (K > 3). We search K.
# Number of "late" recorded values remaining after the 6 early ones:
LATE_COUNT = len(RECORDED) - 1 - len(EARLY_CHECKPOINTS)   # = 5


def solve_segment(start_score, end_score, start_move, n_steps=3,
                  start_level=None, max_level=6):
    """Enumerate all op-sequences that turn start_score into end_score.

    The segment covers moves start_move+1, ..., start_move+n_steps. This is
    the '--solve 2' sub-problem: given two consecutive recorded values, find
    every way the math sequence bridges them.

    Ops per move N:
        same : score += N      (level unchanged)
        up   : score *= N      (level + 1)
        down : score /= N      (level - 1, needs score % N == 0)
    Level stays within [0, max_level]. If start_level is None, we allow any
    starting level in [0, max_level] and report the one each solution used.
    """
    moves = list(range(start_move + 1, start_move + n_steps + 1))
    start_levels = [start_level] if start_level is not None else range(max_level + 1)

    solutions = []
    for lvl0 in start_levels:
        for combo in product(["same", "up", "down"], repeat=n_steps):
            score, level = start_score, lvl0
            trace = []
            ok = True
            for op, N in zip(combo, moves):
                if op == "same":
                    score, level = score + N, level
                elif op == "up":
                    if level >= 2:
                        ok = False
                        break
                    score, level = score * N, level + 1
                else:  # down
                    if score % N != 0 or level - 1 < 0:
                        ok = False
                        break
                    score, level = score // N, level - 1
                if score < 0:
                    ok = False
                    break
                trace.append((N, op, score, level))
            if ok and score == end_score:
                solutions.append((lvl0, trace))
    return moves, solutions


def print_segment(start_score, end_score, start_move, n_steps=3,
                  start_level=None):
    moves, sols = solve_segment(start_score, end_score, start_move, n_steps,
                                start_level)
    print(f"Segment: {start_score} -> {end_score} over moves {moves}"
          + (f", start level {start_level}" if start_level is not None else ""))
    if not sols:
        print("  No op-sequence bridges these two values.\n")
        return
    print(f"  {len(sols)} solution(s):\n")
    for i, (lvl0, trace) in enumerate(sols, 1):
        print(f"  --- solution {i} (start level {lvl0}) ---")
        for N, op, score, level in trace:
            sym = {"same": f"+{N}", "up": f"x{N}", "down": f"/{N}"}[op]
            print(f"    move {N}: {op:<4} {sym:<4} -> score={score}, level={level}")
        print()


def solve_for_K(K, max_moves):
    """Try to find an op sequence for a given late-stride K."""
    # Checkpoint move numbers: early ones + late ones (every K after move 18).
    checkpoints = EARLY_CHECKPOINTS + [18 + K * j for j in range(1, LATE_COUNT + 1)]
    n_moves = checkpoints[-1]
    if n_moves > max_moves:
        return None

    s = Solver()

    # score[i] = score AFTER move i.  score[0] = starting score.
    score = [Int(f"score_{i}") for i in range(n_moves + 1)]
    level = [Int(f"level_{i}") for i in range(n_moves + 1)]

    s += score[0] == 0
    s += level[0] == 0

    for i in range(1, n_moves + 1):
        N = i
        prev, cur = score[i - 1], score[i]

        same = (cur == prev + N)
        up   = (cur == prev * N)
        down = And(prev == cur * N, cur >= 0)   # score divisible by N

        s += Or(same, up, down)
        s += score[i] >= 0

    # tie recorded values to their checkpoint moves
    for idx, mv in enumerate(checkpoints):
        s += score[mv] == RECORDED[idx + 1]

    if s.check() == sat:
        m = s.model()
        scores = [m.evaluate(score[i]).as_long() for i in range(n_moves + 1)]
        levels = [m.evaluate(level[i]).as_long() for i in range(n_moves + 1)]
        return K, checkpoints, scores, levels
    return None


def describe(scores, levels):
    """Reconstruct the op label for each move from consecutive scores."""
    ops = []
    for i in range(1, len(scores)):
        N = i
        prev, cur = scores[i - 1], scores[i]
        if cur == prev + N:
            ops.append(f"move {N:2d}: same  +{N:<3d} -> {cur}")
        elif cur == prev * N:
            ops.append(f"move {N:2d}: UP    x{N:<3d} -> {cur}")
        elif prev == cur * N:
            ops.append(f"move {N:2d}: DOWN  /{N:<3d} -> {cur}")
        else:
            ops.append(f"move {N:2d}: ??? {prev}->{cur}")
    return ops


def solve_first_18():
    """Solve only the first 18 moves (6 early checkpoints). Ignore K and late moves."""
    checkpoints = EARLY_CHECKPOINTS  # [3, 6, 9, 12, 15, 18]
    n_moves = 18

    s = Solver()

    score = [Int(f"score_{i}") for i in range(n_moves + 1)]
    level = [Int(f"level_{i}") for i in range(n_moves + 1)]

    s += score[0] == 0
    s += level[0] == 2          # knight starts two levels up

    for i in range(1, n_moves + 1):
        N = i
        prev, cur = score[i - 1], score[i]
        plev, clev = level[i - 1], level[i]

        # up = climb one level (multiply), down = drop one level (divide),
        # same = stay at this level (add). No hard ceiling on altitude.
        same = And(cur == prev + N, clev == plev)
        up   = And(cur == prev * N, clev == plev + 1)
        down = And(prev == cur * N, cur >= 0, clev == plev - 1)

        s += Or(same, up, down)
        s += level[i] >= 0
        s += score[i] >= 0

    for idx, mv in enumerate(checkpoints):
        s += score[mv] == RECORDED[idx + 1]

    if s.check() == sat:
        m = s.model()
        scores = [m.evaluate(score[i]).as_long() for i in range(n_moves + 1)]
        levels = [m.evaluate(level[i]).as_long() for i in range(n_moves + 1)]
        return scores, levels
    return None


def run_full():
    print("=== Phase 1: solve first 18 moves only ===")
    res18 = solve_first_18()
    if res18 is None:
        print("UNSAT for first 18 moves — check altitude model or recorded values.")
    else:
        scores18, levels18 = res18
        print(f"SAT — scores at checkpoints: {[scores18[mv] for mv in EARLY_CHECKPOINTS]}")
        print(f"Target                     : {RECORDED[1:7]}")
        print()
        for line in describe(scores18, levels18):
            print(line)

    print()
    print("=== Phase 2: search for K (late stride) ===")
    MAX_MOVES = 60
    solution = None
    for K in range(4, 25):
        res = solve_for_K(K, MAX_MOVES)
        if res:
            solution = res
            break

    if solution is None:
        print("No full solution found for any K in range 4-24.")
    else:
        K, checkpoints, scores, levels = solution
        print(f"SOLVED with late-stride K = {K}")
        print(f"Checkpoint moves: {checkpoints}")
        print(f"Recorded target : {RECORDED[1:]}")
        print(f"Scores at those : {[scores[mv] for mv in checkpoints]}")
        print()
        for line in describe(scores, levels):
            print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Knight-score puzzle solver",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--from", dest="frm", type=int,
                        help="segment start score  (omit for full solve)")
    parser.add_argument("--to", type=int,
                        help="segment end score    (omit for full solve)")
    parser.add_argument("--start-move", type=int, default=0,
                        help="move number BEFORE the segment (default 0)")
    parser.add_argument("--steps", type=int, default=3,
                        help="number of moves in the segment (default 3)")
    args = parser.parse_args()

    # hardcoded puzzle constants
    START_SCORE = 0
    START_LEVEL = 2

    if args.frm is not None:
        if args.to is not None:
            # specific target
            print_segment(args.frm, args.to, args.start_move,
                          n_steps=args.steps, start_level=START_LEVEL)
        else:
            # find ALL reachable scores from this start
            moves = list(range(args.start_move + 1, args.start_move + args.steps + 1))
            print(f"All scores reachable from {args.frm} over moves {moves} "
                  f"(start level {START_LEVEL}):\n")
            seen = {}  # end_score -> list of traces
            for lvl0 in range(3):  # levels 0,1,2
                if lvl0 != START_LEVEL:
                    continue
                for combo in product(["same", "up", "down"], repeat=args.steps):
                    score, level = args.frm, lvl0
                    trace = []
                    ok = True
                    for op, N in zip(combo, moves):
                        if op == "same":
                            score, level = score + N, level
                        elif op == "up":
                            if level >= 2:
                                ok = False; break
                            score, level = score * N, level + 1
                        else:
                            if score % N != 0 or level - 1 < 0:
                                ok = False; break
                            score, level = score // N, level - 1
                        if score < 0:
                            ok = False; break
                        trace.append((N, op, score, level))
                    if ok:
                        seen.setdefault(score, []).append((lvl0, trace))

            for end_score in sorted(seen):
                traces = seen[end_score]
                print(f"  -> {end_score}  ({len(traces)} way(s))")
                for lvl0, trace in traces:
                    ops_str = ", ".join(
                        f"+{N}" if op == "same" else f"x{N}" if op == "up" else f"/{N}"
                        for N, op, _, _ in trace
                    )
                    print(f"    level {lvl0}: {ops_str}  => level {trace[-1][3]}")
    else:
        run_full()

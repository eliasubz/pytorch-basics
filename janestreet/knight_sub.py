"""
knight_sub.py — segment path finder with soft targets.

For a given start score, finds all op-sequences of LENGTH moves that land on
any value in a soft target set:
    - all RECORDED values smaller than start_score
    - the two smallest RECORDED values larger than start_score

Level is capped at [0, MAX_LEVEL=2]. 'up' is blocked when already at level 2.
"""

import argparse
from itertools import product

RECORDED = [0, 1, 16, 23, 37, 88, 138, 272, 449, 528, 750, 1100]
MAX_LEVEL = 2


def apply(op, score, level, N):
    if op == "same":
        return score + N, level
    if op == "up":
        if level >= MAX_LEVEL:
            return None            # already at ceiling
        return score * N, level + 1
    if op == "down":
        if score % N != 0 or level - 1 < 0:
            return None
        return score // N, level - 1
    return None


def soft_targets(start_score):
    """All recorded values < start plus the two smallest recorded > start."""
    below = [v for v in RECORDED if v < start_score]
    above = sorted(v for v in RECORDED if v > start_score)
    # return set(below + above)
    return set(RECORDED)


def find_paths(start_score, start_move, steps, start_level):
    targets = soft_targets(start_score)
    moves = list(range(start_move + 1, start_move + steps + 1))
    results = {}   # end_score -> list of traces
    completed_target = []

    for combo in product(["same", "up", "down"], repeat=steps):
        score, level = start_score, start_level
        trace = []
        ok = True
        for op, N in zip(combo, moves):
            res = apply(op, score, level, N)
            if res is None:
                ok = False
                break
            score, level = res
            if score < 0:
                ok = False
                break
            trace.append((N, op, score, level))
        if ok and score in targets:
            results.setdefault(score, []).append(trace)
            completed_target.append(score)

    return moves, targets, results, completed_target


def print_results(start_score, start_move, steps, start_level):
    moves, targets, results, completed_target = find_paths(start_score, start_move, steps, start_level)
    print(f"From {start_score} over moves {moves}  (start level {start_level})")
    print("Start move", start_move)
    print(f"Soft targets: {sorted(targets)}\n")
    end_level = 0
    if not results:
        print("No paths found to any target.")
        return
    for end in sorted(results):
        traces = results[end]
        print(f"  -> {end}  ({len(traces)} path(s))")
        for trace in traces:
            ops = " ".join(
                f"+{N}" if op == "same" else f"x{N}" if op == "up" else f"/{N}"
                for N, op, _, _ in trace
            )
            print(f"    [{ops}]  end level {trace[-1][3]}")
            end_level = trace[-1][3]
    return (moves, targets, results, completed_target, end_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="frm", type=int, required=True,
                        help="start score")
    parser.add_argument("--start-move", type=int, default=0,
                        help="move number before the segment (default 0)")
    parser.add_argument("--steps", type=int, default=3,
                        help="segment length in moves (default 3)")
    args = parser.parse_args()

    r1 = {1: [[(1)]]} 
    completed_target = [0] 

    move_n = 0
    end_lvl = 2
    while True:
        print("#Step: ", move_n )

        found = False
        for t in completed_target:
            RECORDED.remove(t)

            out = print_results(t, move_n, 3, end_lvl)
            if out != None:
                moves, targets, r1, completed_target, end_lvl = out
                print("FOUND: ", moves, targets, r1, completed_target, end_lvl)
                move_n += 3
                found = True
                continue
        if not found: 
            RECORDED.insert(0,t)
            break
            
    rc = RECORDED.copy()
    print(rc)
    new_step_count = 7

    print(f"\nNow trying {7} steps for the new number")
        
    while True:
        print("\n#Step: ", move_n )
        found = False
        for t in completed_target:
            RECORDED.remove(t)

            out = print_results(t, move_n, new_step_count, end_lvl)
            if out != None:
                moves, targets, r1, completed_target, end_lvl = out
                move_n += new_step_count
                found = True
                continue
        if not found: 
            break
print("We did not find the MOVES after: ", moves, targets, r1, completed_target)


# Results The steps cange to 7 and we will now have to verify what steps the knight takes in the field.
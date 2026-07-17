"""
knight_path.py — 3D-knight path finder for the pentomino puzzle.

KEY INSIGHT:
    Genuine 3D knight. Move distributes {0,1,2} across (row, col, altitude).
    Altitude is fixed (floor=1, tower=2). The score op-chain gives dalt per move:
        same  : dalt=0 -> footprint = perms of (1,2)  [knight L]
        up/down dalt=1 -> footprint = (0,±2)/(±2,0)  [2-ortho slide]
        up/down dalt=2 -> footprint = (0,±1)/(±1,0)  [1-ortho step]

ALTITUDE MODEL:
    All squares are at altitude 1 (region base). Towers add 1 → altitude 2.
    13 regions → 13 towers → 13 level-2 cells in the path.
    Knight starts at (7,0) at level 2 (start square has a tower).
    Op-chain levels swing between 1 and 2 only.

PATH LENGTH:
    Op-chain covers moves 1..53 (12 level-2 cells including start=move 0).
    Move 54 must be 'up' (dalt=1) to hit the 13th tower and end the path.
    Total: 54 moves, 55 cells visited, 9 unvisited.

STRATEGY: anchored-segment DFS.
    Chain segments between checkpoint anchors (moves 3,6,9,12,15,18,25,32,39,46,53).
    After move 53, one final 'up' move (move 54) lands on the 13th tower cell.
    Global visited set prevents revisiting. Backtrack across segment boundaries.
"""

from knight_board import CLUES, START, in_bounds

# ---------------------------------------------------------------------------
# Op-chain from knight_sub.py solver output.  (op, level_after_move), move 1..53.
# level before move 1 = 2 (start two levels up).
# ---------------------------------------------------------------------------
CHAIN = [
    ('same', 2), ('same', 2), ('down', 1),                                  # 1-3
    ('same', 1), ('same', 1), ('same', 1),                                  # 4-6
    ('up', 2),   ('down', 1), ('same', 1),                                  # 7-9
    ('same', 1), ('same', 1), ('up', 2),                                    # 10-12
    ('same', 2), ('same', 2), ('down', 1),                                  # 13-15
    ('same', 1), ('same', 1), ('same', 1),                                  # 16-18
    ('same', 1), ('same', 1), ('up', 2), ('same', 2), ('same', 2), ('down', 1), ('same', 1),  # 19-25
    ('same', 1), ('same', 1), ('same', 1), ('same', 1), ('up', 2), ('down', 1), ('same', 1),  # 26-32
    ('up', 2),   ('down', 1), ('same', 1), ('same', 1), ('same', 1), ('same', 1), ('same', 1), # 33-39
    ('same', 1), ('same', 1), ('same', 1), ('same', 1), ('same', 1), ('same', 1), ('same', 1), # 40-46
    ('same', 1), ('same', 1), ('same', 1), ('same', 1), ('same', 1), ('same', 1), ('same', 1), # 47-53
]

# The full score sequence after each move (for filling cells later).
SCORES = [0,
    1, 3, 1,                    # 1-3
    5, 10, 16,                  # 4-6
    112, 14, 23,                # 7-9
    33, 44, 528,                # 10-12
    541, 555, 37,               # 13-15
    53, 70, 88,                 # 16-18
    107, 127, 2667, 2689, 2712, 113, 138,   # 19-25
    164, 191, 219, 248, 7440, 240, 272,     # 26-32
    8976, 264, 299, 335, 372, 410, 449,     # 33-39
    489, 530, 572, 615, 659, 704, 750,      # 40-46
    797, 845, 894, 944, 995, 1047, 1100,    # 47-53
]

TOTAL_MOVES = 54  # 53 known chain moves + 1 final 'up' to 13th tower

# Move 54 is 'up' (dalt=1): the 13th tower visit ending the path.
CHAIN_FULL = CHAIN + [('up', 2)]  # move 54: up, arriving at level 2

# Anchors: written-order checkpoints -> (cell, move).
ANCHORS = [
    (START, 0),
    ((5, 6), 3),   # score 1
    ((4, 4), 6),   # 16
    ((2, 3), 9),   # 23
    ((3, 0), 12),  # 528
    ((0, 5), 15),  # 37
    ((5, 3), 18),  # 88
    ((2, 5), 25),  # 138
    ((5, 5), 32),  # 272
    ((4, 1), 39),  # 449
    ((5, 1), 46),  # 750
    ((0, 7), 53),  # 1100
]


def dalt_per_move():
    """|level change| for each move 1..53 (0 -> knight L, 1 -> 2-slide, 2 -> 1-step)."""
    levels = [2] + [lv for _, lv in CHAIN]
    return [abs(levels[i] - levels[i - 1]) for i in range(1, len(levels))]


def footprints(da):
    """2D (drow,dcol) displacements for a move with altitude change |da|."""
    a, b = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[da]
    fps = set()
    for mr, mc in ((a, b), (b, a)):
        for sr in ([1, -1] if mr else [0]):
            for sc in ([1, -1] if mc else [0]):
                fps.add((sr * mr, sc * mc))
    return sorted(fps)


# Precompute footprint per move index (1..54).
# Moves 1-53 from CHAIN, move 54 is 'up' (dalt=1).
_DALT = dalt_per_move() + [1]  # move 54: dalt=1 (up to tower)
MOVE_FOOTPRINT = {m: footprints(_DALT[m - 1]) for m in range(1, TOTAL_MOVES + 1)}


def neighbors(cell, move_no):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in MOVE_FOOTPRINT[move_no]
            if in_bounds(r + dr, c + dc)]


def _all_tower_regions():
    from knight_board import REGION
    return {REGION[r][c] for r in range(8) for c in range(8)}


def solve():
    """Find the 54-move path hitting all 13 tower regions. Returns 55 cells."""
    from knight_board import REGION
    all_regions = _all_tower_regions()
    path = [START]
    visited = {START}
    # track which region each level-2 visit covers (start cell is level-2, region of START)
    tower_regions_hit = {REGION[START[0]][START[1]]}

    def dfs_to_anchor(cur_move, target_move, target_cell, seg_idx):
        if cur_move == target_move:
            if path[-1] == target_cell:
                return dfs_segment(seg_idx + 1)
            return False
        move_no = cur_move + 1
        lv_after = CHAIN_FULL[move_no - 1][1]
        for nb in neighbors(path[-1], move_no):
            if nb in visited:
                continue
            if move_no == target_move and nb != target_cell:
                continue
            reg = REGION[nb[0]][nb[1]]
            # if this is a tower visit (lv=2), reject if region already has a tower
            if lv_after == 2 and reg in tower_regions_hit:
                continue
            path.append(nb); visited.add(nb)
            if lv_after == 2:
                tower_regions_hit.add(reg)
            if dfs_to_anchor(move_no, target_move, target_cell, seg_idx):
                return True
            path.pop(); visited.discard(nb)
            if lv_after == 2:
                tower_regions_hit.discard(reg)
        return False

    def dfs_segment(seg_idx):
        if seg_idx == len(ANCHORS) - 1:
            # All checkpoints done. Make the final move 54 (up, dalt=1).
            return dfs_final()
        (_, m0), (dst, m1) = ANCHORS[seg_idx], ANCHORS[seg_idx + 1]
        return dfs_to_anchor(m0, m1, dst, seg_idx)

    def dfs_final():
        # Move 54: 'up' (dalt=1). Must land on an unvisited tower cell (lv=2)
        # in a region not yet covered, completing all 13 towers.
        move_no = 54
        for nb in neighbors(path[-1], move_no):
            if nb in visited:
                continue
            reg = REGION[nb[0]][nb[1]]
            if reg in tower_regions_hit:
                continue
            # this visit completes all 13 towers
            if tower_regions_hit | {reg} == all_regions:
                path.append(nb)
                return True
        return False

    if dfs_segment(0):
        return path
    return None


def all_solutions(limit=None):
    """Enumerate distinct valid paths."""
    from knight_board import REGION
    all_regions = _all_tower_regions()
    sols = []
    path = [START]; visited = {START}
    tower_regions_hit = {REGION[START[0]][START[1]]}

    def rec_anchor(cur_move, target_move, target_cell, seg_idx):
        if limit and len(sols) >= limit: return
        if cur_move == target_move:
            if path[-1] == target_cell:
                rec_seg(seg_idx + 1)
            return
        move_no = cur_move + 1
        lv_after = CHAIN_FULL[move_no - 1][1]
        for nb in neighbors(path[-1], move_no):
            if nb in visited: continue
            if move_no == target_move and nb != target_cell: continue
            reg = REGION[nb[0]][nb[1]]
            if lv_after == 2 and reg in tower_regions_hit: continue
            path.append(nb); visited.add(nb)
            if lv_after == 2: tower_regions_hit.add(reg)
            rec_anchor(move_no, target_move, target_cell, seg_idx)
            path.pop(); visited.discard(nb)
            if lv_after == 2: tower_regions_hit.discard(reg)

    def rec_seg(seg_idx):
        if limit and len(sols) >= limit: return
        if seg_idx == len(ANCHORS) - 1:
            rec_final(); return
        (_, m0), (dst, m1) = ANCHORS[seg_idx], ANCHORS[seg_idx + 1]
        rec_anchor(m0, m1, dst, seg_idx)

    def rec_final():
        move_no = 54
        for nb in neighbors(path[-1], move_no):
            if nb in visited: continue
            reg = REGION[nb[0]][nb[1]]
            if reg in tower_regions_hit: continue
            if tower_regions_hit | {reg} == all_regions:
                path.append(nb)
                sols.append(list(path))
                path.pop()

    rec_seg(0)
    return sols


if __name__ == "__main__":
    import sys
    print("Move footprints by altitude change:")
    for da in (0, 1, 2):
        print(f"  dalt={da}: {footprints(da)}")
    print(f"\ndalt sequence (moves 1-53): {dalt_per_move()}")

    if "--count" in sys.argv:
        sols = all_solutions(limit=50)
        print(f"\nDistinct full paths found (capped at 50): {len(sols)}")
        if sols:
            p = sols[0]
            print("First solution cells:", p)
    else:
        path = solve()
        if path is None:
            print("\nNO PATH FOUND.")
        else:
            print(f"\nSOLVED — {len(path)} cells, {len(path)-1} moves.")
            for i, cell in enumerate(path):
                score = SCORES[i] if i < len(SCORES) else f"(move{i})"
                anchor = " <-- CHECKPOINT" if any(cell == a and i == m for a, m in ANCHORS) else ""
                print(f"  move {i:2d}: {cell}  score={score}{anchor}")
            unvisited = [(r, c) for r in range(8) for c in range(8) if (r, c) not in set(path)]
            print(f"\nUnvisited cells ({len(unvisited)}): {unvisited}")

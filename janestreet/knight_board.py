"""
knight_board.py — board encoding + visualizer for the pentomino knight puzzle.

Geometry decoded directly from pent-3-knight-7.png:
  - 8x8 grid, row 0 = TOP, col 0 = LEFT (image coordinates).
  - 13 regions: twelve pentominoes + one 2x2 tetromino (REGION id 3).
  - 12 clue cells with recorded scores.

Model (per puzzle reading):
  - Standard 2D knight moves: a (1,2) L-step. All 64 cells are floor-level.
  - The 'altitude' (same/up/down) is arithmetic-only bookkeeping handled in the
    score solver (knight_sub.py); it imposes NO constraint on which cell the
    knight steps to. Towers are placed after the tour to satisfy the level math.

The knight starts at cell '0' (bottom-left) and makes 60 knight moves,
visiting 61 of the 64 cells; 3 cells remain unvisited.
"""

# ---------------------------------------------------------------------------
# Region map (flood-filled from the image borders). row 0 = top.
# ---------------------------------------------------------------------------
REGION = [
    [0, 0, 0, 0, 0, 1, 1, 1],
    [2, 2, 2, 3, 3, 4, 4, 1],
    [2, 5, 2, 3, 3, 3, 4, 1],
    [6, 5, 5, 7, 7, 8, 4, 4],
    [6, 6, 5, 7, 7, 8, 8, 9],
    [6, 10, 5, 11, 8, 8, 12, 9],
    [6, 10, 11, 11, 11, 12, 12, 9],
    [10, 10, 10, 11, 12, 12, 9, 9],
]

# ---------------------------------------------------------------------------
# Clue cells: (row, col) -> recorded score.  row 0 = top.
# ---------------------------------------------------------------------------
CLUES = {
    (0, 5): 37,
    (0, 7): 1100,
    (2, 3): 23,
    (2, 5): 138,
    (3, 0): 528,
    (4, 1): 449,
    (4, 4): 16,
    (5, 1): 750,
    (5, 3): 88,
    (5, 5): 272,
    (5, 6): 1,
    (7, 0): 0,
}

START = (7, 0)          # the '0' cell
N_ROWS = N_COLS = 8

# Recorded scores in the order the knight wrote them (checkpoints).
# Solved value-order chain from knight_sub.py:
RECORDED_SORTED = [0, 1, 16, 23, 37, 88, 138, 272, 449, 528, 750, 1100]

# Knight move offsets (2D L-step).
KNIGHT_MOVES = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
]


def in_bounds(r, c):
    return 0 <= r < N_ROWS and 0 <= c < N_COLS


def knight_neighbors(r, c):
    """All in-bounds cells reachable by one knight move from (r, c)."""
    out = []
    for dr, dc in KNIGHT_MOVES:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc):
            out.append((nr, nc))
    return out


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------
def _region_borders():
    """Return sets of (r,c) edges that are region borders, for drawing.
    v_border[(r,c)] True -> thick line on the RIGHT of cell (r,c).
    h_border[(r,c)] True -> thick line on the BOTTOM of cell (r,c).
    """
    v_border, h_border = {}, {}
    for r in range(N_ROWS):
        for c in range(N_COLS):
            if c < N_COLS - 1:
                v_border[(r, c)] = REGION[r][c] != REGION[r][c + 1]
            if r < N_ROWS - 1:
                h_border[(r, c)] = REGION[r][c] != REGION[r + 1][c]
    return v_border, h_border


def draw(path=None, scores=None, unvisited=None, title="pentomino knight",
         save=None, show=True):
    """Render the board.

    path      : list of (r,c) cells in visit order (draws the knight's route).
    scores    : dict (r,c) -> score to print in cells (defaults to CLUES).
    unvisited : iterable of (r,c) cells to shade (the never-visited squares).
    save      : filepath to write a PNG.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if scores is None:
        scores = CLUES
    unvisited = set(unvisited or [])

    fig, ax = plt.subplots(figsize=(8, 8))
    # We plot in image coords: x = col, y = row, row 0 at top -> invert y.
    ax.set_xlim(-0.5, N_COLS - 0.5)
    ax.set_ylim(N_ROWS - 0.5, -0.5)     # inverted so row 0 is on top
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    # shade unvisited
    for (r, c) in unvisited:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                               facecolor="#ffd6d6", edgecolor="none", zorder=0))

    # thin grid
    for r in range(N_ROWS + 1):
        ax.plot([-0.5, N_COLS - 0.5], [r - 0.5, r - 0.5], color="0.75", lw=0.8, zorder=1)
    for c in range(N_COLS + 1):
        ax.plot([c - 0.5, c - 0.5], [-0.5, N_ROWS - 0.5], color="0.75", lw=0.8, zorder=1)

    # thick region borders
    v_border, h_border = _region_borders()
    for (r, c), on in v_border.items():
        if on:
            ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color="black", lw=3, zorder=2)
    for (r, c), on in h_border.items():
        if on:
            ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color="black", lw=3, zorder=2)
    # outer frame
    ax.add_patch(Rectangle((-0.5, -0.5), N_COLS, N_ROWS, fill=False,
                           edgecolor="black", lw=4, zorder=2))

    # path
    if path:
        xs = [c for (r, c) in path]
        ys = [r for (r, c) in path]
        ax.plot(xs, ys, "-", color="#1f77b4", lw=1.8, alpha=0.7, zorder=3)
        ax.plot(xs[0], ys[0], "o", color="green", ms=12, zorder=4)   # start
        ax.plot(xs[-1], ys[-1], "s", color="red", ms=10, zorder=4)   # end
        for i, (r, c) in enumerate(path):
            ax.text(c + 0.28, r - 0.28, str(i), fontsize=6, color="#1f77b4",
                    ha="center", va="center", zorder=5)

    # scores / clues
    for (r, c), val in scores.items():
        is_clue = (r, c) in CLUES
        ax.text(c, r, str(val), fontsize=13 if is_clue else 8,
                fontweight="bold" if is_clue else "normal",
                color="black" if is_clue else "#444",
                ha="center", va="center", zorder=6)

    ax.set_title(title)
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
        print(f"saved -> {save}")
    if show:
        plt.show()
    plt.close(fig)


def print_ascii(scores=None):
    """Quick terminal view: region letters + clue values."""
    if scores is None:
        scores = CLUES
    print("   " + "".join(f" c{c} " for c in range(N_COLS)))
    for r in range(N_ROWS):
        cells = []
        for c in range(N_COLS):
            if (r, c) in scores:
                cells.append(f"{scores[(r,c)]:>3}")
            else:
                cells.append(f" {chr(ord('A') + REGION[r][c])} ")
        print(f"r{r} " + " ".join(cells))
    print("\n(letters = region id A..M; numbers = clues)")


if __name__ == "__main__":
    import sys
    print(f"Regions: {max(max(row) for row in REGION) + 1}  "
          f"(expected 13: 12 pentominoes + 1 tetromino)\n")
    print_ascii()
    out = "board_check.png"
    if "--no-show" in sys.argv:
        draw(save=out, show=False, title="decoded board — verify vs pent-3-knight-7.png")
    else:
        draw(save=out, title="decoded board — verify vs pent-3-knight-7.png")

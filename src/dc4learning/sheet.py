"""Sheet geometry and drawing (meters, DC4 simulator coordinates).

Orientation matches the DC3 notebooks (`ax.set_ylim(336, 0)`, `j = 93 - x/0.0254`):
delivery runs down the page and +x is on the left, i.e. viewed from above.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

TEE_Y = 38.405
HOG_Y = 32.004
HOUSE_R = 1.829
BACK_Y = TEE_Y + HOUSE_R
STONE_R = 0.145
SHEET_HALF_W = 4.75 / 2

TEAM_COLOR = {"team0": "#d1495b", "team1": "#e8b100"}  # red / yellow
EDGE = "#333333"

OTHER_TEAM = {"team0": "team1", "team1": "team0"}


def stones_xy(stone_json: dict, team: str) -> list[tuple[int, float, float]]:
    """(index, x, y) of stones in play; (0, 0) means not thrown / removed."""
    pts = [(s["x"], s["y"]) for s in (stone_json.get(team) or [])]
    return [(i, x, y) for i, (x, y) in enumerate(pts) if not (x == 0.0 and y == 0.0)]


def draw_sheet(ax, ymin: float = HOG_Y - 0.6, ymax: float = BACK_Y + 0.6):
    ax.set_facecolor("white")
    for r, c in [(1.829, "#cfe0f5"), (1.219, "white"), (0.610, "#f5d0d0"), (0.152, "white")]:
        ax.add_patch(plt.Circle((0, TEE_Y), r, facecolor=c, edgecolor="grey", lw=0.4, zorder=0))
    ax.plot((0, 0), (ymin, ymax), lw=0.4, c="grey", zorder=1)  # center line
    for y, lw in [(TEE_Y, 0.4), (HOG_Y, 1.2), (BACK_Y, 0.4)]:
        ax.plot((-SHEET_HALF_W, SHEET_HALF_W), (y, y), lw=lw, c="grey", zorder=1)
    ax.set_xlim(SHEET_HALF_W, -SHEET_HALF_W)  # +x on the left (top-down view)
    ax.set_ylim(ymax, ymin)  # delivery runs downward
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_stones(ax, stone_json: dict, label: bool = True, highlight: tuple[str, int] | None = None):
    for team in ["team0", "team1"]:
        for i, x, y in stones_xy(stone_json, team):
            lw = 1.8 if highlight == (team, i) else 0.6
            ax.add_patch(plt.Circle((x, y), STONE_R, facecolor=TEAM_COLOR[team], edgecolor=EDGE, lw=lw, zorder=3))
            if label:
                ax.text(x, y, str(i), ha="center", va="center", fontsize=6, color=EDGE, zorder=4)


def changed_stone(prev_json: dict | None, cur_json: dict) -> tuple[str, int] | None:
    """The stone that appeared from (0, 0), i.e. the stone just thrown."""
    if prev_json is None:
        return None
    for team in ["team0", "team1"]:
        for i, (p, c) in enumerate(zip(prev_json.get(team) or [], cur_json.get(team) or [])):
            if p["x"] == 0 and p["y"] == 0 and not (c["x"] == 0 and c["y"] == 0):
                return team, i
    return None


# --- DC3-style pixel grid (training input), same as PositionToPixel in convert_log.ipynb ---

GRID_SHAPE = (27 * 12 + 12, 12 * 15 + 7)


def position_to_pixel(x: float, y: float) -> tuple[int, int]:
    i = int(np.round((y - HOG_Y) / 0.0254))
    j = 93 - int(np.round(x / 0.0254))
    return i, j


def to_sheet_grid(stone_json: dict, hammer: str) -> np.ndarray:
    """(2, H, W) occupancy grid. Channel 0 = non-hammer, 1 = hammer."""
    sheet = np.zeros((2, *GRID_SHAPE), dtype="int")
    for ch, team in enumerate([OTHER_TEAM[hammer], hammer]):
        for _, x, y in stones_xy(stone_json, team):
            i, j = position_to_pixel(x, y)
            if 0 <= i < GRID_SHAPE[0] and 0 <= j < GRID_SHAPE[1]:
                sheet[ch, i, j] = 1
    return sheet

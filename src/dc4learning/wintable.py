"""Win table for the C++ client, estimated from downloaded self-play games.

    python -m dc4learning.wintable --model gen0      # games where both teams ran gen0
    python -m dc4learning.wintable --since 2026-11-10

Writes tables/win_table-<selection>-n<matches>-<hash>.csv under `config.data_dir()`, with a sidecar
.json (the matches, the versions that played them, the counts behind every cell, and the commit of
this code; see dc4learning.naming). Link the client's win_table.csv to it to use it.

Layout, as `skip.cpp` reads it (19 x 11): table[lead + 9][column] is the probability that the team
holding the hammer at the start of an end wins, where lead is that team's score minus the other's
(clipped to -9..9) and column is the end in a 10-end game (10 - standard ends + end number). Column 10
is the state after the last standard end: a tie goes to an extra end (estimated from extra ends), any
lead has decided the game (1 or 0).

Games lost on time are left out: their result reflects the clock, not the position.

As in the DC3 notebooks (training_light_fixsize.ipynb), a cell is the observed win rate where it has
at least --min-games games and neither all wins nor all losses; otherwise a default: 0 for a deficit,
1 for a lead or a tie, nudged by 0.001 per point towards 0.5.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import data
from .config import data_dir
from .naming import finalize

LEADS = 9
COLUMNS = 11


def default_table() -> np.ndarray:
    base = np.concatenate([np.zeros(LEADS), np.ones(LEADS + 1)])
    for i in range(LEADS):
        base[i] += i * 1e-3
        base[-i - 1] -= i * 1e-3
    table = np.repeat(base[:, None], COLUMNS, axis=1)
    table[:LEADS, -1] = 0.0  # behind after the last end: lost
    table[LEADS + 1 :, -1] = 1.0  # ahead: won
    return table


def counts(index) -> tuple[np.ndarray, np.ndarray]:
    """(games, hammer-team wins) per (lead + 9, column)."""
    games = np.zeros((2 * LEADS + 1, COLUMNS), dtype=int)
    wins = np.zeros_like(games)
    for match_id in index["match_id"]:
        meta, states = data.load_match(match_id)
        match = meta["match"]
        by_id = {match["first_team_id"]: "team0", match["second_team_id"]: "team1"}
        s0, s1, n_std = meta["score0"], meta["score1"], meta["standard_end_count"]
        if meta["winner"] is None:
            continue
        for end in range(len(s0)):
            first = states[(states["end_number"] == end) & (states["total_shot_number"] == 0)]
            if first.empty:
                continue
            hammer = "team1" if by_id[first.iloc[-1]["next_shot_team_id"]] == "team0" else "team0"
            lead0 = sum(s0[:end]) - sum(s1[:end])
            lead = int(np.clip(lead0 if hammer == "team0" else -lead0, -LEADS, LEADS))
            col = COLUMNS - 1 if end >= n_std else COLUMNS - 1 - n_std + end
            if col < 0:
                continue
            games[lead + LEADS, col] += 1
            wins[lead + LEADS, col] += meta["winner"] == hammer
    return games, wins


def build(selection: str | None = None, min_games: int = 1, **filters) -> tuple:
    # Games decided by a side running out of time say nothing about who was winning on the sheet.
    index = data.select_matches(data.games(since=filters.get("since")), **filters, include_time_losses=False)
    games, wins = counts(index)
    table = default_table()
    observed = (games >= min_games) & (wins > 0) & (wins < games)
    table[observed] = wins[observed] / games[observed]
    selection = selection or data.selection_label("selfplay", **filters)
    tmp = data_dir() / "tables" / f".{selection}.tmp.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(tmp, table, delimiter=",")
    out = finalize(
        tmp,
        f"win_table-{selection}-n{len(index)}",
        ".csv",
        {
            "kind": "win_table",
            "params": {"min_games": min_games, **filters},
            "observed_cells": int(observed.sum()),
            "games": games.tolist(),
            "wins": wins.tolist(),
            "played_by": data.played_by(index),
            "match_ids": index["match_id"].tolist(),
        },
    )
    return out, table, games


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="only games where both teams used this model (name or sha256 prefix)")
    p.add_argument("--client", help="only games where both teams used this client commit")
    p.add_argument("--image", help="only games where both teams used this image tag")
    p.add_argument("--since", help="only games started at or after this time")
    p.add_argument("--selection", help="name part before -n<matches> (default: from the filters)")
    p.add_argument("--min-games", type=int, default=1, help="games a cell needs to use its win rate")
    args = p.parse_args(argv)
    out, table, games = build(
        args.selection, args.min_games, image=args.image, model=args.model, client=args.client, since=args.since
    )
    print(f"{out}  ({int((games > 0).sum())} cells with games)", file=sys.stderr)
    for lead in range(LEADS, -LEADS - 1, -1):
        row = " ".join(f"{v:4.2f}" for v in table[lead + LEADS])
        print(f"{lead:+3d}: {row}   n={json.dumps(games[lead + LEADS].tolist())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

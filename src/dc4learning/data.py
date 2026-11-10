"""Self-play games on the server, and training datasets built from them.

    python -m dc4learning.data build --model gen0    # dataset from games where both teams ran gen0
    python -m dc4learning.data build --since 2026-11-10T10:03

The games are read from the server whenever they're needed (`games`, `load_match`); nothing the
server's database already holds is kept here. What is kept, under `config.data_dir()` (default
<repo>/data, not committed): datasets/<selection>-n<matches>-<hash>.npz (+ .json), see `build_dataset`.

Match names written by dc4learning.selfplay carry each team's version ID,
<client commit>.<model>.<model hash> (older matches: the image tag):
    <prefix>-<version>-<timestamp>-m<n>                  both teams the same version
    <prefix>-<version0>-vs-<version1>-<timestamp>-m<n>   team0 = version0 (throws first in end 1)
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import versions
from .api import DC4Client
from .config import data_dir
from .encoding import score_class, shot_to_action, stone_pixels
from .naming import finalize, slug

NAME_RE = re.compile(r"^(?P<prefix>.+?)-(?:(?P<img0>[^-]+)-vs-(?P<img1>[^-]+)|(?P<img>[^-]+))-\d{8}-\d{6}-m\d+$")


def images_from_name(name: str) -> tuple[str | None, str | None]:
    """(team0 image, team1 image) from a self-play match name; (None, None) for older names."""
    m = NAME_RE.match(name or "")
    if not m or (m["img"] and m["img"].isdigit()):  # "selfplay-20261109-..." has no image
        return None, None
    return (m["img0"], m["img1"]) if m["img0"] else (m["img"], m["img"])


def summarize(match: dict, last: pd.Series, ends_played: int) -> dict:
    """Per-match summary: teams, score by end, winner and clocks (from the final state)."""
    by_id = {match["first_team_id"]: "team0", match["second_team_id"]: "team1"}
    score = match.get("score") or {}
    s0 = list(score.get("team0") or [])[:ends_played]
    s1 = list(score.get("team1") or [])[:ends_played]
    return {
        "match_id": match["match_id"],
        "match_name": match["match_name"],
        "tournament_name": (match.get("tournament") or {}).get("tournament_name"),
        "started_at": match["started_at"],
        "finished_at": last["created_at"],
        "standard_end_count": match["standard_end_count"],
        "time_limit": match["time_limit"],
        "ends_played": ends_played,
        "extra_ends": max(0, ends_played - match["standard_end_count"]),
        "score0": s0,
        "score1": s1,
        "total0": int(sum(s0)),
        "total1": int(sum(s1)),
        "winner": by_id.get(last["winner_team_id"]),
        "time_left0": float(last["first_team_remaining_time"]),
        "time_left1": float(last["second_team_remaining_time"]),
    }


_client: DC4Client | None = None
_summaries: dict[str, dict] = {}  # finished matches never change: cached for this process only


def _server() -> DC4Client:
    global _client
    _client = _client or DC4Client()
    return _client


def _summary(match_id: str) -> dict | None:
    """Summary of a finished match from the server (the match and its last end only); None while
    it is still being played."""
    if match_id in _summaries:
        return _summaries[match_id]
    c = _server()
    ends = c.get(f"/matches/{match_id}/ends")
    if not ends:
        return None
    last_end = max(ends)
    states = c.get(f"/matches/{match_id}/ends/{last_end}/states")
    if not states:
        return None
    last = pd.Series(max(states, key=lambda st: (st["total_shot_number"], st["created_at"])))
    if last["winner_team_id"] is None:
        return None
    summary = summarize(c.get_match(match_id), last, last_end + 1)
    _summaries[match_id] = summary
    return summary


def games(prefix: str = "selfplay-", since: str | None = None, workers: int = 8) -> pd.DataFrame:
    """Finished matches whose name starts with `prefix`, read from the server: one row per match
    with its summary, each team's versions (dc4learning.versions) and the flags below.

    Nothing is stored locally: the server's database is the record of the games."""
    listed = _server().list_matches()
    if listed.empty:
        return listed
    listed = listed[listed["match_name"].fillna("").str.startswith(prefix)]
    if since:
        listed = listed[pd.to_datetime(listed["started_at"]) >= pd.Timestamp(since)]
    ids = listed["match_id"].tolist()

    def fetch(match_id):
        try:
            return _summary(match_id)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            print(f"{match_id}: skipped: {e}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(workers) as pool:
        rows = [r for r in pool.map(fetch, ids) if r]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ("image", "image_id", "client", "model", "model_sha256"):
        for t in (0, 1):
            out[f"{col}{t}"] = None
    for i, row in out.iterrows():
        for t, v in enumerate(versions.match_versions(row["match_id"], row["match_name"]).values()):
            for col in ("image_id", "client", "model", "model_sha256"):
                out.at[i, f"{col}{t}"] = v[col]
            out.at[i, f"image{t}"] = (v["image"] or "").rsplit(":", 1)[-1] or images_from_name(row["match_name"])[t]
    # time_loss: one side's clock ran out, so the result was decided by time, not play; the
    # unfinished end has no score (match_samples skips it). interrupted: the self-play record
    # marks the match as cut short by an outage; left out everywhere by default.
    out["time_loss"] = (out["time_left0"] <= 0) | (out["time_left1"] <= 0)
    out["interrupted"] = [(versions.match_record(m) or {}).get("status") == "interrupted" for m in out["match_id"]]
    out["started_at"] = pd.to_datetime(out["started_at"])
    out["finished_at"] = pd.to_datetime(out["finished_at"])
    return out.sort_values("started_at", ignore_index=True)


def load_match(match_id: str) -> tuple[dict, pd.DataFrame]:
    """(summary with the match JSON under "match", states) of a finished match, from the server."""
    c = _server()
    states = c.load_states(match_id)
    meta = dict(_summary(match_id) or {})
    meta["match"] = c.get_match(match_id)
    return meta, states


def match_samples(meta: dict, states: pd.DataFrame) -> dict[str, np.ndarray]:
    """Training samples of one match: one per shot thrown, from the position it was thrown from.

    policy = the shot the client chose (`actual_*`, the requested shot before execution noise),
    score = the hammer team's net points in that end.
    """
    match = meta["match"]
    by_id = {match["first_team_id"]: "team0", match["second_team_id"]: "team1"}
    other = {"team0": "team1", "team1": "team0"}
    s0, s1 = meta["score0"], meta["score1"]
    out = {k: [] for k in ("pixels", "shot", "policy", "score", "end", "team", "won")}
    for end, g in states.groupby("end_number"):
        first = g[g["total_shot_number"] == 0]
        if first.empty or end >= len(s0) or not (g["total_shot_number"] == 16).any():
            continue  # unfinished end (e.g. a side ran out of time): its score was never recorded
        hammer = other[by_id[first.iloc[-1]["next_shot_team_id"]]]
        net = (s0[end] - s1[end]) * (1 if hammer == "team0" else -1)
        thrown = g[g["actual_translational_velocity"].notna() & (g["total_shot_number"] < 16)]
        for _, row in thrown.drop_duplicates("total_shot_number", keep="last").iterrows():
            action = shot_to_action(
                row["actual_translational_velocity"], row["actual_angular_velocity"], row["actual_shot_angle"]
            )
            team = by_id.get(row["shot_team_id"])
            out["pixels"].append(stone_pixels(row["stones"], hammer))
            out["shot"].append(row["total_shot_number"])
            out["policy"].append(action)
            out["score"].append(score_class(net))
            out["end"].append(end)
            out["team"].append(0 if team == "team0" else 1)
            out["won"].append(meta["winner"] == team)
    return out


def select_matches(
    index: pd.DataFrame,
    image: str | None = None,
    model: str | None = None,
    client: str | None = None,
    since: str | None = None,
    include_interrupted: bool = False,
    include_time_losses: bool = True,
) -> pd.DataFrame:
    """Matches where both teams match the filters: `model` is a model name or sha256 prefix,
    `client` a client commit, `image` an image tag; `since` a start time.
    Interrupted matches are left out, and games lost on time too if `include_time_losses` is False
    (see `games`)."""
    if not include_interrupted and "interrupted" in index:
        index = index[~index["interrupted"].astype(bool)]
    if not include_time_losses and "time_loss" in index:
        index = index[~index["time_loss"].astype(bool)]
    if image:
        index = index[(index["image0"] == image) & (index["image1"] == image)]
    if client:
        index = index[(index["client0"] == client) & (index["client1"] == client)]
    if model:

        def is_model(t):
            return (index[f"model{t}"] == model) | index[f"model_sha256{t}"].fillna("").str.startswith(model)

        index = index[is_model(0) & is_model(1)]
    if since:
        index = index[index["started_at"] >= pd.Timestamp(since)]
    return index


def selection_label(prefix: str, image=None, model=None, client=None, since=None) -> str:
    """The <selection> part of dataset and win table names, e.g. selfplay-gen0."""
    parts = [prefix, model, client and f"client-{client}", image and f"image-{image}", since and f"since-{since}"]
    return slug("-".join(p for p in parts if p))


def played_by(index: pd.DataFrame) -> dict[str, int]:
    """Number of matches per pairing of versions, e.g. {"a604b03/gen0 vs a604b03/gen0": 506}."""
    v = (
        index["client0"].fillna("?")
        + "/"
        + index["model0"].fillna("?")
        + " vs "
        + index["client1"].fillna("?")
        + "/"
        + index["model1"].fillna("?")
    )
    return v.value_counts().to_dict()


def build_dataset(
    selection: str | None = None,
    image: str | None = None,
    since: str | None = None,
    root: Path | None = None,
    model: str | None = None,
    client: str | None = None,
) -> Path:
    """Samples of finished matches (read from the server) where both teams match the filters.

    Writes datasets/<selection>-n<matches>-<hash>.npz and its sidecar .json: the filters, the
    match IDs, the versions that played them, and the commit of this code (dc4learning.naming).
    """
    root = root or data_dir()
    index = select_matches(games(since=since), image, model, client)
    selection = slug(selection or selection_label("selfplay", image, model, client, since))
    cols = {k: [] for k in ("pixels", "shot", "policy", "score", "end", "team", "won", "match")}
    for i, match_id in enumerate(index["match_id"]):
        meta, states = load_match(match_id)
        s = match_samples(meta, states)
        for k, v in s.items():
            cols[k].extend(v)
        cols["match"].extend([i] * len(s["shot"]))
    arrays = {
        "pixels": np.array(cols["pixels"], dtype=np.int8).reshape(-1, 2, 8, 2),
        "shot": np.array(cols["shot"], dtype=np.int8),
        "policy": np.array(cols["policy"], dtype=np.int16),
        "score": np.array(cols["score"], dtype=np.int8),
        "end": np.array(cols["end"], dtype=np.int8),
        "team": np.array(cols["team"], dtype=np.int8),
        "won": np.array(cols["won"], dtype=bool),
        "match": np.array(cols["match"], dtype=np.int32),
        "match_ids": index["match_id"].to_numpy(dtype=str),
    }
    tmp = root / "datasets" / f".{selection}.tmp.npz"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tmp, **arrays)
    out = finalize(
        tmp,
        f"{selection}-n{len(index)}",
        ".npz",
        {
            "kind": "dataset",
            "params": {"image": image, "model": model, "client": client, "since": since},
            "samples": len(arrays["shot"]),
            "played_by": played_by(index),
            "match_ids": index["match_id"].tolist(),
        },
    )
    off_grid = int((arrays["policy"] < 0).sum())
    print(
        f"{out}: {len(arrays['shot'])} samples from {len(index)} matches"
        f" ({off_grid} shots off the policy grid, kept without a policy target)",
        file=sys.stderr,
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build a training dataset from finished matches on the server")
    b.add_argument("--selection", default=None, help="name part before -n<matches> (default: from the filters)")
    b.add_argument("--image", help="only matches where both teams used this client image (tag)")
    b.add_argument("--model", help="only matches where both teams used this model (name or sha256 prefix)")
    b.add_argument("--client", help="only matches where both teams used this client commit")
    b.add_argument("--since", help="only matches started at or after this time, e.g. 2026-11-09T02:30")
    args = p.parse_args(argv)
    build_dataset(args.selection, args.image, args.since, model=args.model, client=args.client)
    return 0


if __name__ == "__main__":
    sys.exit(main())

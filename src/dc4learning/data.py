"""Self-play logs: download finished matches from the server, and build training datasets.

    python -m dc4learning.data sync                  # download new finished matches
    python -m dc4learning.data build                 # dataset from all downloaded matches
    python -m dc4learning.data build --image <tag>   # only matches played by that client image

Layout under `config.data_dir()` (default <repo>/data, not committed):
    matches/<match_id>.parquet   one row per state (dc4learning.api.DC4Client.load_states)
    matches/<match_id>.json      match summary (see `summarize`)
    matches.parquet              all summaries, one row per match
    datasets/<name>.npz          training samples (see `build_dataset`)

Match names written by dc4learning.selfplay carry the client images:
    <prefix>-<image>-<timestamp>-m<n>            both teams the same image
    <prefix>-<image0>-vs-<image1>-<timestamp>-m<n>   team0 = image0 (throws first in end 1)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import versions
from .api import DC4Client
from .config import data_dir
from .encoding import score_class, shot_to_action, stone_pixels

NAME_RE = re.compile(r"^(?P<prefix>.+?)-(?:(?P<img0>[^-]+)-vs-(?P<img1>[^-]+)|(?P<img>[^-]+))-\d{8}-\d{6}-m\d+$")


def images_from_name(name: str) -> tuple[str | None, str | None]:
    """(team0 image, team1 image) from a self-play match name; (None, None) for older names."""
    m = NAME_RE.match(name or "")
    if not m or (m["img"] and m["img"].isdigit()):  # "selfplay-20261109-..." has no image
        return None, None
    return (m["img0"], m["img1"]) if m["img0"] else (m["img"], m["img"])


def summarize(match: dict, states: pd.DataFrame) -> dict:
    """Per-match summary: teams, images, score by end, winner (from the final state)."""
    by_id = {match["first_team_id"]: "team0", match["second_team_id"]: "team1"}
    score = match.get("score") or {}
    last = states.iloc[-1]
    ends_played = int(states["end_number"].max()) + 1
    s0 = list(score.get("team0") or [])[:ends_played]
    s1 = list(score.get("team1") or [])[:ends_played]
    img0, img1 = images_from_name(match["match_name"])
    return {
        "match_id": match["match_id"],
        "match_name": match["match_name"],
        "started_at": match["started_at"],
        "finished_at": last["created_at"],
        "image0": img0,
        "image1": img1,
        "standard_end_count": match["standard_end_count"],
        "time_limit": match["time_limit"],
        "ends_played": ends_played,
        "extra_ends": max(0, ends_played - match["standard_end_count"]),
        "score0": s0,
        "score1": s1,
        "total0": int(sum(s0)),
        "total1": int(sum(s1)),
        "winner": by_id.get(last["winner_team_id"]),
        "n_shots": int(states["actual_translational_velocity"].notna().sum()),
        "time_left0": float(last["first_team_remaining_time"]),
        "time_left1": float(last["second_team_remaining_time"]),
    }


def sync(prefix: str = "selfplay-", client: DC4Client | None = None, root: Path | None = None) -> pd.DataFrame:
    """Download finished matches whose name starts with `prefix` and aren't downloaded yet."""
    client = client or DC4Client()
    root = root or data_dir()
    mdir = root / "matches"
    mdir.mkdir(parents=True, exist_ok=True)
    listed = client.list_matches()
    listed = listed[listed["match_name"].fillna("").str.startswith(prefix)]
    new = 0
    for match_id in listed["match_id"]:
        if (mdir / f"{match_id}.json").exists():
            continue
        try:
            states = client.load_states(match_id)
            if states.empty or pd.isna(states.iloc[-1]["winner_team_id"]):
                continue  # still being played
            match = client.get_match(match_id)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            print(f"{match_id}: skipped, will retry next sync: {e}", file=sys.stderr)
            continue
        out = states.copy()
        out["stones"] = out["stones"].map(json.dumps)
        out.to_parquet(mdir / f"{match_id}.parquet", index=False)
        (mdir / f"{match_id}.json").write_text(json.dumps({**summarize(match, states), "match": match}, default=str))
        new += 1
    index = rebuild_index(root)
    print(f"{new} new matches downloaded, {len(index)} in {root}", file=sys.stderr)
    return index


def rebuild_index(root: Path | None = None) -> pd.DataFrame:
    root = root or data_dir()
    rows = []
    for f in sorted((root / "matches").glob("*.json")):
        d = json.loads(f.read_text())
        d.pop("match", None)
        rows.append(d)
    index = pd.DataFrame(rows)
    if not index.empty:
        # Versions are looked up on every rebuild, so names given to models later show up here.
        for col in ("image", "image_id", "client", "model", "model_sha256"):
            for t in (0, 1):
                index[f"{col}{t}"] = None
        for i, row in index.iterrows():
            for t, v in enumerate(versions.match_versions(row["match_id"], row["match_name"]).values()):
                for col in ("image_id", "client", "model", "model_sha256"):
                    index.at[i, f"{col}{t}"] = v[col]
                tag = (v["image"] or "").rsplit(":", 1)[-1] or images_from_name(row["match_name"])[t]
                index.at[i, f"image{t}"] = tag
        index["started_at"] = pd.to_datetime(index["started_at"])
        index["finished_at"] = pd.to_datetime(index["finished_at"])
        index = index.sort_values("started_at", ignore_index=True)
    index.to_parquet(root / "matches.parquet", index=False)
    return index


def load_index(root: Path | None = None) -> pd.DataFrame:
    root = root or data_dir()
    path = root / "matches.parquet"
    return pd.read_parquet(path) if path.exists() else rebuild_index(root)


def load_match(match_id: str, root: Path | None = None) -> tuple[dict, pd.DataFrame]:
    """(match JSON, states) of a downloaded match; `stones` back as dicts."""
    root = root or data_dir()
    meta = json.loads((root / "matches" / f"{match_id}.json").read_text())
    states = pd.read_parquet(root / "matches" / f"{match_id}.parquet")
    states["stones"] = states["stones"].map(json.loads)
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
        if first.empty or end >= len(s0):
            continue
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


def build_dataset(
    name: str = "dataset",
    image: str | None = None,
    since: str | None = None,
    root: Path | None = None,
    model: str | None = None,
    client: str | None = None,
) -> Path:
    """Samples of downloaded matches where both teams match the filters.

    `model` is a model name or sha256 prefix, `client` a client commit, `image` an image tag.
    Writes <name>.npz and <name>.json: the filters, the matches, and the versions that played them.
    """
    root = root or data_dir()
    index = rebuild_index(root)  # current version names
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
    cols = {k: [] for k in ("pixels", "shot", "policy", "score", "end", "team", "won", "match")}
    for i, match_id in enumerate(index["match_id"]):
        meta, states = load_match(match_id, root)
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
    out = root / "datasets" / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    played_by = (
        index.assign(
            v=lambda d: (
                d["client0"].fillna("?")
                + "/"
                + d["model0"].fillna("?")
                + " vs "
                + d["client1"].fillna("?")
                + "/"
                + d["model1"].fillna("?")
            )
        )["v"]
        .value_counts()
        .to_dict()
    )
    meta = {
        "name": name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "filters": {"image": image, "model": model, "client": client, "since": since},
        "samples": len(arrays["shot"]),
        "matches": len(index),
        "played_by": played_by,
        "match_ids": index["match_id"].tolist(),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
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
    s = sub.add_parser("sync", help="download new finished matches")
    s.add_argument("--prefix", default="selfplay-", help="match name prefix")
    b = sub.add_parser("build", help="build a training dataset from downloaded matches")
    b.add_argument("--name", default=None, help="dataset name (default: dataset-<image or all>-<date>)")
    b.add_argument("--image", help="only matches where both teams used this client image (tag)")
    b.add_argument("--model", help="only matches where both teams used this model (name or sha256 prefix)")
    b.add_argument("--client", help="only matches where both teams used this client commit")
    b.add_argument("--since", help="only matches started at or after this time, e.g. 2026-11-09T02:30")
    args = p.parse_args(argv)
    if args.cmd == "sync":
        sync(args.prefix)
    else:
        label = args.model or args.client or args.image or "all"
        name = args.name or f"dataset-{label}-{datetime.now():%Y%m%d-%H%M}"
        build_dataset(name, args.image, args.since, model=args.model, client=args.client)
    return 0


if __name__ == "__main__":
    sys.exit(main())

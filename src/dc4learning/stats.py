"""Self-play status and results.

    python -m dc4learning.stats                         # all self-play games on the server
    python -m dc4learning.stats --since 2026-11-10T10:03
    watch -n 300 python -m dc4learning.stats            # refresh every 5 minutes

Reports the matches running on the cluster (when SELFPLAY_NAMESPACE is set and kubectl works),
throughput, and per pairing of versions (client commit / model, see dc4learning.versions): games,
wins, the first team's win rate, and for two different versions a two-sided binomial test of "the two are equally strong", with each image
playing both sides (as in the DC3 si2025 evaluation).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime

import pandas as pd

from .config import load_env
from .data import games


def cluster_status() -> str | None:
    namespace = os.environ.get("SELFPLAY_NAMESPACE")
    if not namespace:
        return None
    try:
        out = subprocess.run(
            ["kubectl", "-n", namespace, "get", "jobs", "-l", "dc4learning/match-id", "-o", "json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return f"cluster: kubectl failed ({e.__class__.__name__})"
    jobs = json.loads(out)["items"]
    now = datetime.now(UTC)
    lines = []
    by_match: dict[str, list] = {}
    for j in jobs:
        by_match.setdefault(j["metadata"]["labels"]["dc4learning/match-id"], []).append(j)
    for match_id, js in sorted(by_match.items()):
        start = min(datetime.fromisoformat(j["metadata"]["creationTimestamp"]) for j in js)
        images = ", ".join(sorted({j["spec"]["template"]["spec"]["containers"][0]["image"] for j in js}))
        states = []
        for j in js:
            conds = {c["type"] for c in j.get("status", {}).get("conditions") or [] if c["status"] == "True"}
            states.append("failed" if "Failed" in conds else "done" if "Complete" in conds else "running")
        lines.append(f"  {match_id}  {int((now - start).total_seconds() // 60):3d} min  {'/'.join(states)}  {images}")
    return f"cluster: {len(by_match)} matches in {namespace}\n" + "\n".join(lines)


def pairing_table(index: pd.DataFrame) -> pd.DataFrame:
    """One row per unordered pair of versions (client/model): games, wins of A and B, draws, A's win rate, p-value."""
    from scipy.stats import binomtest

    rows = {}

    def version(m, t):
        # client commit / model, e.g. "a604b03/gen0"; the image tag if the versions are unknown
        if m.get(f"client{t}") or m.get(f"model{t}"):
            return f"{m.get(f'client{t}') or '?'}/{m.get(f'model{t}') or '?'}"
        return m.get(f"image{t}") or "?"

    for _, m in index.iterrows():
        a, b = version(m, 0), version(m, 1)
        key = tuple(sorted((a, b)))
        r = rows.setdefault(
            key,
            {
                "A": key[0],
                "B": key[1],
                "games": 0,
                "A wins": 0,
                "B wins": 0,
                "draws": 0,
                "first team wins": 0,
                "time losses": 0,
            },
        )
        r["games"] += 1
        r["time losses"] += bool(m.get("time_loss"))
        if m["winner"] is None:
            r["draws"] += 1
            continue
        winner_image = a if m["winner"] == "team0" else b
        r["first team wins"] += m["winner"] == "team0"
        if a == b:
            continue
        r["A wins" if winner_image == key[0] else "B wins"] += 1
    out = pd.DataFrame(rows.values())
    if out.empty:
        return out
    decided = out["games"] - out["draws"]
    out["first team win rate"] = (out["first team wins"] / decided).round(3)
    out["A win rate"] = (out["A wins"] / decided).where(out["A"] != out["B"]).round(3)
    out["p (A = B)"] = [
        binomtest(int(r["A wins"]), int(r["A wins"] + r["B wins"])).pvalue
        if r["A"] != r["B"] and r["A wins"] + r["B wins"]
        else None
        for _, r in out.iterrows()
    ]
    return out.drop(columns=["first team wins"])


def report(index: pd.DataFrame) -> str:
    if index.empty:
        return "no finished matches downloaded yet"
    excluded = int(index["interrupted"].sum()) if "interrupted" in index else 0
    index = index[~index["interrupted"].astype(bool)] if "interrupted" in index else index
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)  # the server's times are naive UTC
    finished = index["finished_at"]
    last_hour = (finished >= now - pd.Timedelta(hours=1)).sum()
    durations = (index["finished_at"] - index["started_at"]).dt.total_seconds() / 60
    lines = [
        f"matches: {len(index)} finished, {last_hour} in the last hour, last at {finished.max():%Y-%m-%d %H:%M} UTC"
        + (f" ({excluded} interrupted, left out)" if excluded else ""),
        f"time losses: {index['time_loss'].mean():.0%} of games ended with a side out of time"
        if "time_loss" in index
        else "",
        (
            f"game: {index['ends_played'].mean():.2f} ends on average, {(index['extra_ends'] > 0).mean():.1%} with an extra end,"
            f" {durations.median():.1f} min median, {(index['winner'].isna()).sum()} draws"
        ),
        "",
        pairing_table(index).to_string(index=False),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="selfplay-", help="match name prefix")
    p.add_argument("--since", help="only matches started at or after this time (UTC), e.g. 2026-11-10T10:03")
    args = p.parse_args(argv)
    load_env()
    index = games(args.prefix, args.since)
    status = cluster_status()
    if status:
        print(status + "\n")
    print(report(index))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Read-only client for the DigitalCurling4 server REST API.

Only the GET endpoints are used; they need no authentication.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from .config import load_env

SHOT_COLS = [
    "team_id",
    "translational_velocity",
    "angular_velocity",
    "shot_angle",
    "actual_translational_velocity",
    "actual_angular_velocity",
    "actual_shot_angle",
]


class DC4Client:
    def __init__(self, base_url: str | None = None, timeout: float = 30, retries: int = 4):
        load_env()
        base_url = base_url or os.environ.get("DC_SERVER_URL")
        if not base_url:
            raise ValueError("Pass base_url or set DC_SERVER_URL (environment or the repo's .env)")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def get(self, path: str, **params):
        """GET and decode JSON. Server errors (5xx) and dropped connections are retried with backoff."""
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code < 500 or attempt == self.retries:
                    raise
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == self.retries:
                    raise
            time.sleep(min(30, 2**attempt))

    def list_matches(self) -> pd.DataFrame:
        """All matches in all tournaments, newest first."""
        names = sorted({t["tournament_name"] for t in self.get("/tournaments")})
        rows = [m for name in names for m in self.get("/matches", tournament_name=name)]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.drop_duplicates("match_id").sort_values("started_at", ascending=False, ignore_index=True)

    def get_match(self, match_id: str) -> dict:
        return self.get(f"/matches/{match_id}")

    def load_states(self, match_id: str) -> pd.DataFrame:
        """One row per state (sheet position) of the match.

        Each shot has `pre_shot_state_id` (state it was thrown from) and `post_shot_state_id`
        (state it produced). Each row carries the shot thrown *from* the state (plain columns,
        team in `shot_team_id`) and the shot that led *to* it (`prev_*`, team in `prev_shot_team_id`).
        """
        rows = []
        for end in self.get(f"/matches/{match_id}/ends"):
            end_shots = self.get(f"/matches/{match_id}/ends/{end}/shots")
            shots_by_pre = {s["pre_shot_state_id"]: s for s in end_shots}
            shots_by_post = {s["post_shot_state_id"]: s for s in end_shots}
            for st in self.get(f"/matches/{match_id}/ends/{end}/states"):
                nxt = shots_by_pre.get(st["state_id"], {})
                prv = shots_by_post.get(st["state_id"], {})
                score = st.get("score") or {}
                rows.append(
                    {
                        "state_id": st["state_id"],
                        "end_number": st["end_number"],
                        "team_shot_number": st["team_shot_number"],
                        "total_shot_number": st["total_shot_number"],
                        "created_at": st["created_at"],
                        "next_shot_team_id": st["next_shot_team_id"],
                        "winner_team_id": st["winner_team_id"],
                        "first_team_remaining_time": st["first_team_remaining_time"],
                        "second_team_remaining_time": st["second_team_remaining_time"],
                        "stones": st["stone_coordinate"]["data"],
                        "score_team0": score.get("team0"),
                        "score_team1": score.get("team1"),
                        **{("shot_" + c if c == "team_id" else c): nxt.get(c) for c in SHOT_COLS},
                        **{("prev_shot_" + c if c == "team_id" else "prev_" + c): prv.get(c) for c in SHOT_COLS},
                    }
                )

        states = pd.DataFrame(rows)
        if states.empty:
            return states
        states = states.sort_values(["end_number", "total_shot_number", "created_at"], ignore_index=True)
        for c in SHOT_COLS[1:]:
            states[c] = states[c].astype(float)
            states["prev_" + c] = states["prev_" + c].astype(float)
        return states


def team_maps(match: dict) -> tuple[dict, dict]:
    """team0 = first_team (throws first in end 0), team1 = second_team."""
    names = {"team0": match["first_team_name"], "team1": match["second_team_name"]}
    by_id = {match["first_team_id"]: "team0", match["second_team_id"]: "team1"}
    return names, by_id

"""File names that record where a model, dataset or win table came from.

Every artifact ends in the first 8 hex digits of its own sha256, and names its inputs by theirs, so
the chain can be followed by file name alone (`find -name '*-<hash>.*'`):

    model     <gen>-e<epoch>-p<parent>-d<dataset>-<hash>.pt   gen1-e007-p588f425c-d1a2b3c4d-3f2a9c1d.pt
    dataset   <selection>-n<matches>-<hash>.npz                selfplay-gen0-n506-1a2b3c4d.npz
    win table win_table-<selection>-n<matches>-<hash>.csv     win_table-selfplay-gen0-n506-7c9e0f12.csv
    root      <gen>-<hash>.pt, win_table-<hash>.csv           no recorded origin (e.g. gen0-588f425c.pt)

Next to each artifact is a sidecar with the same name and `.json`: kind, sha256, creation time, the
commit of this repo that made it (`code`), its inputs, and its parameters (for a dataset or win
table, the match IDs). Consumers use links with the names they expect (traced.pt, win_table.csv,
data/models/best.pt); the versioned file is the original.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .config import repo_root

HASH = 8

MODEL_RE = re.compile(
    r"^(?P<gen>[a-z0-9]+)-e(?P<epoch>\d+)-p(?P<parent>[0-9a-f]{8})-d(?P<dataset>[0-9a-f]{8})-(?P<hash>[0-9a-f]{8})\.pt$"
)
ROOT_MODEL_RE = re.compile(r"^(?P<gen>[a-z0-9]+)-(?P<hash>[0-9a-f]{8})\.pt$")
DATASET_RE = re.compile(r"^(?P<selection>.+)-n(?P<matches>\d+)-(?P<hash>[0-9a-f]{8})\.npz$")
TABLE_RE = re.compile(r"^win_table-(?:(?P<selection>.+)-n(?P<matches>\d+)-)?(?P<hash>[0-9a-f]{8})\.csv$")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_version() -> dict:
    """Commit of this repo and whether tracked files had uncommitted changes."""
    root = repo_root()
    # safe.directory: in the dev container the checkout belongs to another user; these calls only read.
    git = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    try:
        commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            [*git, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(dirty)}


def slug(text: str) -> str:
    """Lowercase alphanumerics and dashes, for the selection part of a name."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "all"


def next_gen(parent_name: str | None) -> str:
    """gen<N+1> from a parent named gen<N>-...; gen0 without a parent."""
    m = re.match(r"^gen(\d+)-", parent_name or "")
    return f"gen{int(m[1]) + 1}" if m else "gen0" if parent_name is None else "gen1"


def finalize(tmp: Path, name_without_hash: str, suffix: str, sidecar: dict) -> Path:
    """Rename `tmp` to `<name_without_hash>-<hash><suffix>` and write its sidecar; return the new path."""
    digest = sha256(tmp)
    path = tmp.with_name(f"{name_without_hash}-{digest[:HASH]}{suffix}")
    tmp.replace(path)
    meta = {
        "file": path.name,
        "sha256": digest,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(),
        **sidecar,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return path


def parse(name: str) -> dict | None:
    """The fields of a versioned file name, with `kind`; None if it doesn't follow the convention."""
    for kind, rx in (("model", MODEL_RE), ("model", ROOT_MODEL_RE), ("dataset", DATASET_RE), ("win_table", TABLE_RE)):
        m = rx.match(name)
        if m:
            return {"kind": kind, **{k: v for k, v in m.groupdict().items() if v is not None}}
    return None


def find(hash8: str, roots: list[Path]) -> Path | None:
    """The artifact whose name ends in -<hash8>.<ext> (not a sidecar), searching `roots` recursively."""
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob(f"*-{hash8}.*"):
            if p.suffix != ".json" and not p.is_symlink():
                return p
    return None


def trace(path: str | Path, roots: list[Path], depth: int = 0) -> list[str]:
    """Lines describing an artifact and, recursively, its parent model and dataset."""
    path = Path(path).resolve()
    pad = "  " * depth
    fields = parse(path.name)
    if not fields:
        return [f"{pad}{path.name}: not a versioned name"]
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    actual = sha256(path)[:HASH] if path.exists() else None
    check = "" if actual in (None, fields["hash"]) else f"  (content hash is {actual}: file was changed!)"
    code = meta.get("code") or {}
    code_s = f" code {code['commit'][:8]}{'+dirty' if code.get('dirty') else ''}" if code.get("commit") else ""
    lines = [f"{pad}{path.name}  [{fields['kind']}{code_s}]{check}"]
    if meta.get("params"):
        lines.append(f"{pad}  params: {json.dumps(meta['params'])}")
    if fields["kind"] in ("dataset", "win_table") and meta:
        lines.append(f"{pad}  {len(meta.get('match_ids', []))} matches, played by {meta.get('played_by')}")
    if not meta and depth == 0:
        lines.append(f"{pad}  no sidecar {meta_path.name}")
    for key, label in (("parent", "parent model"), ("dataset", "trained on")):
        if key in fields:
            found = find(fields[key], roots)
            lines.append(f"{pad}  {label}: " + ("" if found else f"*-{fields[key]}.* not found"))
            if found:
                lines.extend(trace(found, roots, depth + 2))
    return lines

"""Which client and model played each match, and where each model came from.

Three identities:
  - image: the Docker image ID (sha256 of its config). The cluster reports the same ID for the
    image a pod actually ran (`containerStatuses[].imageID`), so it can't drift from the tag.
  - client: the DigitalCurling4-CNN commit the client binary was built from.
  - model: sha256 of the `traced.pt` inside the image. The CNN repo doesn't track its model
    files, so the commit alone doesn't say which model an image runs.

Model files: a versioned original, `<generation>-<first 8 hex digits of sha256>.pt` (e.g.
`gen0-588f425c.pt`), and links with the names its users expect: `traced.pt` in the CNN repo,
`data/models/best.pt` (the current champion) here. Promoting a model is repointing the link.

Records, under `config.data_dir()/versions` (not committed):
    images.json            image ID -> refs, client commit, model sha256, built from
    models.json            model sha256 -> name, source, parent model, training dataset
    matches/<id>.json      per match: each team's image (ref and the ID the pod ran)

Images built with scripts/client-image.sh or scripts/model-image.sh carry the same facts as labels
(`LABEL_*`), so `register-image` can read them back from any copy of the image.

    python -m dc4learning.versions list                         # images and models
    python -m dc4learning.versions show <match_id>              # what played a match
    python -m dc4learning.versions register-image cnn-client:<tag> [--client-commit C] [--model-name N]
    python -m dc4learning.versions register-model runs/<run>/traced-NNN.pt [--name N]
    python -m dc4learning.versions name-model <sha256 prefix> <name>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .config import data_dir

LABEL_CLIENT = "org.opencontainers.image.revision"
LABEL_MODEL_SHA = "io.github.dc4learning.model.sha256"
LABEL_MODEL_NAME = "io.github.dc4learning.model.name"
LABEL_MODEL_SOURCE = "io.github.dc4learning.model.source"
LABEL_PARENT_IMAGE = "io.github.dc4learning.parent-image"

MODEL_PATH_IN_IMAGE = "/app/src/engine/model/traced.pt"


def _dir() -> Path:
    d = data_dir() / "versions"
    (d / "matches").mkdir(parents=True, exist_ok=True)
    return d


def _load(name: str) -> dict:
    path = _dir() / name
    return json.loads(path.read_text()) if path.exists() else {}


def _save(name: str, obj: dict) -> None:
    path = _dir() / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    tmp.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def images() -> dict:
    return _load("images.json")


def models() -> dict:
    return _load("models.json")


def model_label(sha: str | None) -> str | None:
    """Model name if it has one, else the first 8 hex digits of its sha256."""
    if not sha:
        return None
    return models().get(sha, {}).get("name") or sha[:8]


def image_by_ref(ref: str) -> tuple[str, dict] | tuple[None, None]:
    """(image ID, record) of the image last registered under `ref` (e.g. cnn-client:a604b03)."""
    found = [(i, r) for i, r in images().items() if ref in r.get("refs", [])]
    if not found:
        return None, None
    return max(found, key=lambda x: x[1].get("registered_at", ""))


def register_model(sha: str, **info) -> dict:
    """Add or update a model record; only non-empty fields are written."""
    ms = models()
    rec = ms.get(sha, {"registered_at": _now()})
    rec.update({k: v for k, v in info.items() if v not in (None, "")})
    ms[sha] = rec
    _save("models.json", ms)
    return rec


def register_trained_model(path: str | Path, name: str | None = None) -> tuple[str, dict]:
    """Record a `traced.pt`; if it is a training export (runs/<run>/traced-NNN.pt), with its lineage."""
    path = Path(path)
    sha = file_sha256(path)
    real = path.resolve()  # e.g. data/models/best.pt -> gen1-1a2b3c4d.pt: record the versioned file
    info: dict = {"name": name, "source": str(real)}
    run_info_path = real.parent / "args.json"
    if run_info_path.exists():
        run = json.loads(run_info_path.read_text())
        info["run"] = path.parent.name
        info["epoch"] = path.stem.rsplit("-", 1)[-1]
        info["parent"] = run.get("init_sha256")
        info["init"] = run.get("init")
        info["dataset"] = run.get("dataset")
    return sha, register_model(sha, **info)


def register_image(
    ref: str,
    client_commit: str | None = None,
    model_name: str | None = None,
    model_source: str | None = None,
) -> tuple[str, dict]:
    """Record a local Docker image: its ID, labels, and the sha256 of its model."""
    inspect = json.loads(
        subprocess.run(["docker", "image", "inspect", ref], capture_output=True, text=True, check=True).stdout
    )[0]
    image_id = inspect["Id"]
    labels = inspect["Config"].get("Labels") or {}
    model_sha = labels.get(LABEL_MODEL_SHA)
    if not model_sha:
        out = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "sha256sum", ref, MODEL_PATH_IN_IMAGE],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        model_sha = out.split()[0]
    imgs = images()
    rec = imgs.get(image_id, {"registered_at": _now(), "refs": []})
    if ref not in rec["refs"]:
        rec["refs"].append(ref)
    parent = labels.get(LABEL_PARENT_IMAGE)
    if not (client_commit or labels.get(LABEL_CLIENT)) and parent:
        client_commit = (image_by_ref(parent)[1] or {}).get("client_commit")  # same binary as its parent
    rec.update(
        {
            "client_commit": client_commit or labels.get(LABEL_CLIENT) or rec.get("client_commit"),
            "model_sha256": model_sha,
            "parent_image": labels.get(LABEL_PARENT_IMAGE) or rec.get("parent_image"),
            "created": inspect.get("Created"),
        }
    )
    imgs[image_id] = rec
    _save("images.json", imgs)
    register_model(
        model_sha,
        name=model_name or labels.get(LABEL_MODEL_NAME),
        source=model_source or labels.get(LABEL_MODEL_SOURCE),
    )
    return image_id, rec


def team_version(ref: str | None, image_id: str | None) -> dict:
    """Client commit and model of a team, from the image ID the pod ran (else from the ref)."""
    rec = images().get(image_id or "") if image_id else None
    if rec is None and ref:
        image_id_by_ref, rec = image_by_ref(ref)
        image_id = image_id or image_id_by_ref
    rec = rec or {}
    sha = rec.get("model_sha256")
    return {
        "image": ref,
        "image_id": image_id,
        "client": rec.get("client_commit"),
        "model_sha256": sha,
        "model": model_label(sha),
    }


# --- per-match records, written by dc4learning.selfplay ---


def _merge(into: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(into.get(k), dict):
            _merge(into[k], v)
        else:
            into[k] = v
    return into


def record_match(match_id: str, **info) -> None:
    """Add to a match's record; nested dicts (e.g. teams) are merged, not replaced."""
    path = _dir() / "matches" / f"{match_id}.json"
    rec = json.loads(path.read_text()) if path.exists() else {"match_id": match_id}
    path.write_text(json.dumps(_merge(rec, info), indent=1, sort_keys=True))


def match_record(match_id: str) -> dict | None:
    path = _dir() / "matches" / f"{match_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def match_versions(match_id: str, match_name: str | None = None) -> dict[str, dict]:
    """{"team0": team_version, "team1": team_version} for a match.

    From the match record if dc4learning.selfplay wrote one, else from the image tags in the
    match name (resolved through images.json), else unknown.
    """
    rec = match_record(match_id) or {}
    teams = rec.get("teams") or {}
    out = {}
    for t in ("team0", "team1"):
        team = teams.get(t) or {}
        ref, image_id = team.get("image"), team.get("image_id")
        if not ref and match_name:
            from .data import images_from_name

            tag = images_from_name(match_name)[0 if t == "team0" else 1]
            if tag:
                ref = next((r for rec in images().values() for r in rec["refs"] if r.endswith(":" + tag)), None)
        out[t] = team_version(ref, image_id)
    return out


# --- CLI ---


def _list() -> None:
    ms = models()
    print("models:")
    for sha, m in sorted(ms.items(), key=lambda x: x[1].get("registered_at", "")):
        parent = model_label(m.get("parent")) if m.get("parent") else "-"
        print(f"  {sha[:12]}  {m.get('name') or '-':16s} parent {parent:16s} {m.get('source') or ''}")
        if m.get("dataset"):
            d = m["dataset"]
            print(f"  {'':12s}  trained on {d.get('name')}: {d.get('matches')} matches by {d.get('images')}")
    print("images:")
    for image_id, r in sorted(images().items(), key=lambda x: x[1].get("registered_at", "")):
        print(
            f"  {image_id[7:19]}  {', '.join(r['refs']):32s} client {r.get('client_commit') or '?':10s}"
            f" model {model_label(r.get('model_sha256'))}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="images and models")
    s = sub.add_parser("show", help="what played a match")
    s.add_argument("match_id")
    r = sub.add_parser("register-image", help="record a local Docker image")
    r.add_argument("ref")
    r.add_argument("--client-commit")
    r.add_argument("--model-name")
    r.add_argument("--model-source")
    m = sub.add_parser("register-model", help="record a traced.pt, with its training lineage if it has one")
    m.add_argument("path")
    m.add_argument("--name")
    n = sub.add_parser("name-model", help="give a model a short name")
    n.add_argument("sha_prefix")
    n.add_argument("name")
    args = p.parse_args(argv)
    if args.cmd == "list":
        _list()
    elif args.cmd == "show":
        print(json.dumps({"record": match_record(args.match_id), "versions": match_versions(args.match_id)}, indent=1))
    elif args.cmd == "register-image":
        image_id, rec = register_image(args.ref, args.client_commit, args.model_name, args.model_source)
        print(f"{args.ref}: {image_id[:19]} client {rec.get('client_commit')} model {model_label(rec['model_sha256'])}")
    elif args.cmd == "register-model":
        sha, rec = register_trained_model(args.path, args.name)
        print(f"{sha} {rec.get('name') or ''}")
    else:
        matches = [sha for sha in models() if sha.startswith(args.sha_prefix)]
        if len(matches) != 1:
            p.error(f"{len(matches)} models match {args.sha_prefix}")
        register_model(matches[0], name=args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

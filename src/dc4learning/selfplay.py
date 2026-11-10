"""Self-play on the cluster: keep N matches running, start a new one when one finishes.

Each match is one match created on the server (POST /matches) plus two Kubernetes Jobs,
one CNN client per team. Every client requests `--gpu-per-client` GPU slots (time-sliced
`nvidia.com/gpu`), so the GPU node's slot count also caps how many clients fit at once.

All state lives in the cluster as Job labels, so the loop can be stopped and restarted
without losing track of running matches.

    python -m dc4learning.selfplay --parallel 2

Cluster-specific settings (namespace, server URL, Secret names, image) have no defaults:
pass them as options or set them in the repo's `.env` (not committed; see `.env.example`).

No credentials pass through this machine: matches are created by a short Job that logs in
with `--creator-secret`, and the clients use `--team0-secret` / `--team1-secret` (Secrets
with keys `auth-id`, `auth-pw`, already in the namespace). This process only needs kubectl
access to the namespace.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime

from . import versions
from .config import load_env

SELFPLAY_LABEL = "dc4learning/selfplay"
CREATOR_LABEL = "dc4learning/create-match"
MATCH_LABEL = "dc4learning/match-id"
TEAM_LABEL = "dc4learning/team"


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------- server


# The password goes to curl through a config on stdin, so it is not in any command line.
CREATE_MATCH_SCRIPT = """\
printf 'user = "%s:%s"\\n' "$DC_AUTH_ID" "$DC_AUTH_PW" \\
  | curl -fsS -m 30 -K - -X POST "$DC_HOST/matches" \\
      -H 'Content-Type: application/json' -H 'Accept: application/json' --data "$PAYLOAD"
"""


def create_match(args: argparse.Namespace, name: str) -> str:
    """Create a match from inside the cluster with a one-shot Job; return its ID."""
    payload = {
        "game_mode": args.game_mode,
        "tournament": {"tournament_name": args.tournament},
        "simulator": {"simulator_name": args.simulator},
        "applied_rule": args.applied_rule,
        "time_limit": args.time_limit,
        "extra_end_time_limit": args.extra_end_time_limit,
        "standard_end_count": args.end_count,
        "match_name": name,
        "positioned_stones_pattern": args.positioned_pattern,
    }
    # A valid Job name: lowercase alphanumerics and dashes, at most 63 characters.
    job_name = "create-" + re.sub(r"[^a-z0-9-]+", "-", name.lower())[-50:].strip("-")
    labels = {SELFPLAY_LABEL + "-creator": args.run_label, CREATOR_LABEL: "true"}
    secret = args.creator_secret
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "labels": labels},
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 120,
            "ttlSecondsAfterFinished": 600,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 10001,
                        "runAsGroup": 10001,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "containers": [
                        {
                            "name": "create-match",
                            "image": args.curl_image,
                            "command": ["sh", "-c", CREATE_MATCH_SCRIPT],
                            "env": [
                                {"name": "DC_HOST", "value": args.server_url.rstrip("/")},
                                {"name": "PAYLOAD", "value": json.dumps(payload)},
                                {
                                    "name": "DC_AUTH_ID",
                                    "valueFrom": {"secretKeyRef": {"name": secret, "key": "auth-id"}},
                                },
                                {
                                    "name": "DC_AUTH_PW",
                                    "valueFrom": {"secretKeyRef": {"name": secret, "key": "auth-pw"}},
                                },
                            ],
                            "resources": {"requests": {"cpu": "10m", "memory": "16Mi"}, "limits": {"memory": "64Mi"}},
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                "capabilities": {"drop": ["ALL"]},
                            },
                        }
                    ],
                },
            },
        },
    }
    if args.dry_run:
        kubectl(args, "create", "-f", "-", stdin=json.dumps(job))
        return "dry-run-0000-0000-000000000000"

    kubectl(args, "create", "-f", "-", stdin=json.dumps(job))
    try:
        deadline = time.monotonic() + 150
        while True:
            status = json.loads(kubectl(args, "get", "job", job_name, "-o", "json")).get("status", {})
            if status.get("succeeded") or status.get("failed"):
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f"Job {job_name} did not finish")
            time.sleep(2)
        out = kubectl(args, "logs", f"job/{job_name}").strip()
        if not status.get("succeeded"):
            raise RuntimeError(f"Job {job_name} failed: {out}")
        return str(json.loads(out))
    finally:
        kubectl(args, "delete", "job", job_name, "--wait=false", "--ignore-not-found")


# --------------------------------------------------------------------------- kubectl


def kubectl(args: argparse.Namespace, *cmd: str, stdin: str | None = None) -> str:
    full = ["kubectl", "-n", args.namespace, *cmd]
    if args.dry_run and cmd[0] in ("create", "delete"):
        log("dry-run: " + " ".join(full))
        if stdin:
            print(stdin)
        return ""
    return subprocess.run(full, input=stdin, capture_output=True, text=True, check=True).stdout


def image_tag(image: str) -> str:
    """Short label of an image for match names: its tag (or name), without dashes."""
    tag = image.rsplit("/", 1)[-1]
    tag = tag.split(":", 1)[1] if ":" in tag else tag
    return tag.replace("-", "_")


def match_images(args: argparse.Namespace, n: int) -> tuple[str, str]:
    """(team0 image, team1 image) of match n. With an opponent, the sides alternate."""
    if not args.opponent_image:
        return args.image, args.image
    return (args.image, args.opponent_image) if n % 2 else (args.opponent_image, args.image)


def version_label(image: str) -> str:
    """What a match name says about a team: its version ID (`<client>.<model>.<model hash>`,
    from the version records) or, for an unrecorded image, its tag."""
    return versions.version_id(image) or image_tag(image)


def match_name(args: argparse.Namespace, n: int) -> str:
    """<prefix>-<version>-<ts>-m<n>, or <prefix>-<team0 version>-vs-<team1 version>-<ts>-m<n>.

    The server stores the name, so its database records what played each match; dc4learning.data
    and dc4learning.versions read it back."""
    a, b = (version_label(i) for i in match_images(args, n))
    teams = a if a == b else f"{a}-vs-{b}"
    return f"{args.match_prefix}-{teams}-{datetime.now():%Y%m%d-%H%M%S}-m{n}"


def default_tournament(args: argparse.Namespace) -> str:
    """One tournament per run: selfplay-<version>, or eval-<candidate>-vs-<opponent> (sides alternate)."""
    if args.opponent_image:
        return f"eval-{version_label(args.image)}-vs-{version_label(args.opponent_image)}"
    return f"{args.match_prefix}-{version_label(args.image)}"


def job_manifest(args: argparse.Namespace, match_id: str, match_name: str, team: int, image: str) -> dict:
    labels = {
        "app.kubernetes.io/name": "cnn-client",
        "app.kubernetes.io/component": "client",
        SELFPLAY_LABEL: args.run_label,
        MATCH_LABEL: match_id,
        TEAM_LABEL: str(team),
    }
    secret = args.team0_secret if team == 0 else args.team1_secret
    env = [
        {"name": "DC_HOST", "value": args.client_host or args.server_url},
        {"name": "DC_MATCH_ID", "value": match_id},
        {"name": "DC_TEAM", "value": str(team)},
        {"name": "DC_AUTH_ID", "valueFrom": {"secretKeyRef": {"name": secret, "key": "auth-id"}}},
        {"name": "DC_AUTH_PW", "valueFrom": {"secretKeyRef": {"name": secret, "key": "auth-pw"}}},
    ]
    for item in filter(None, (args.client_env or "").split(",")):
        key, _, value = item.strip().partition("=")
        env.append({"name": key, "value": value})
    if args.threads:
        env.append({"name": "DC_CLIENT_THREADS", "value": str(args.threads)})
    else:
        # Same as the CNN deploy base: threads = CPU limit, i.e. all node CPUs with no limit.
        env.append(
            {
                "name": "DC_CLIENT_THREADS",
                "valueFrom": {
                    "resourceFieldRef": {"containerName": "client", "resource": "limits.cpu", "divisor": "1"}
                },
            }
        )
    mounts = [{"name": "tmp", "mountPath": "/tmp"}]
    volumes: list[dict] = [{"name": "tmp", "emptyDir": {}}]
    if args.ca_configmap:
        env.append({"name": "SSL_CERT_FILE", "value": "/etc/dc4/ca.crt"})
        mounts.append({"name": "server-ca", "mountPath": "/etc/dc4", "readOnly": True})
        volumes.append({"name": "server-ca", "configMap": {"name": args.ca_configmap}})

    pod_spec: dict = {
        "restartPolicy": "OnFailure",
        "automountServiceAccountToken": False,
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 10001,
            "runAsGroup": 10001,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "containers": [
            {
                "name": "client",
                "image": image,
                "imagePullPolicy": args.pull_policy,
                "env": env,
                "resources": {
                    "requests": {"cpu": "1", "memory": "2Gi"},
                    "limits": {"memory": "6Gi", "nvidia.com/gpu": args.gpu_per_client},
                },
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "readOnlyRootFilesystem": True,
                    "capabilities": {"drop": ["ALL"]},
                },
                "volumeMounts": mounts,
            }
        ],
        "volumes": volumes,
    }
    if args.runtime_class:
        pod_spec["runtimeClassName"] = args.runtime_class
    if args.image_pull_secret:
        pod_spec["imagePullSecrets"] = [{"name": args.image_pull_secret}]

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"selfplay-{match_id.split('-')[-1]}-t{team}",
            "labels": labels,
            "annotations": {"dc4learning/match-name": match_name},
        },
        "spec": {
            "backoffLimit": 3,
            "activeDeadlineSeconds": args.deadline,
            "ttlSecondsAfterFinished": 3600,
            "template": {"metadata": {"labels": labels}, "spec": pod_spec},
        },
    }


def list_matches(args: argparse.Namespace) -> dict[str, list[dict]]:
    """Jobs of this run, grouped by match ID."""
    out = kubectl(args, "get", "jobs", "-l", f"{SELFPLAY_LABEL}={args.run_label}", "-o", "json")
    by_match: dict[str, list[dict]] = {}
    for job in json.loads(out)["items"]:
        by_match.setdefault(job["metadata"]["labels"][MATCH_LABEL], []).append(job)
    return by_match


def job_state(job: dict) -> str:
    for c in job.get("status", {}).get("conditions") or []:
        if c["status"] == "True" and c["type"] in ("Complete", "Failed"):
            return c["type"]
    return "Running"


# --------------------------------------------------------------------------- loop


def start_match(args: argparse.Namespace, n: int) -> None:
    name = match_name(args, n)
    images = match_images(args, n)
    match_id = create_match(args, name)
    if args.verbose or args.dry_run:
        log(f"match {n}: created {name} ({match_id})")
    # Nothing is stored locally per match: the server's match and tournament names carry each
    # team's version ID (see version_label), and its database holds everything else.
    for team in (0, 1):
        manifest = job_manifest(args, match_id, name, team, images[team])
        kubectl(args, "create", "-f", "-", stdin=json.dumps(manifest))


def run(args: argparse.Namespace) -> int:
    stop = False

    def request_stop(*_):
        nonlocal stop
        if stop:
            sys.exit(130)
        stop = True
        log("stopping: no new matches; running ones keep going (Ctrl-C again to quit now)")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    started = finished = failed = consecutive_failures = 0
    while True:
        matches = {} if args.dry_run else list_matches(args)
        running = 0
        for match_id, jobs in matches.items():
            states = [job_state(j) for j in jobs]
            if "Failed" in states or (len(jobs) == 2 and all(s == "Complete" for s in states)):
                # Done, or one side gave up (the other would only wait for its deadline).
                ok = "Failed" not in states
                finished += ok
                failed += not ok
                consecutive_failures = 0 if ok else consecutive_failures + 1
                teams = ", ".join(f"t{j['metadata']['labels'][TEAM_LABEL]}={s}" for j, s in zip(jobs, states))
                if args.verbose or not ok:  # the server records every finished game; log only failures
                    log(f"match {match_id} {'finished' if ok else 'FAILED'} ({teams}); {finished} ok, {failed} failed")
                kubectl(args, "delete", "jobs", "-l", f"{MATCH_LABEL}={match_id}", "--wait=false")
            else:
                running += 1

        if consecutive_failures >= args.max_consecutive_failures:
            log(f"{consecutive_failures} failures in a row; stopping. Check `kubectl -n {args.namespace} logs`.")
            return 1

        while not stop and running < args.parallel and (args.matches == 0 or started < args.matches):
            started += 1
            try:
                start_match(args, started)
            except (RuntimeError, ValueError, subprocess.CalledProcessError) as e:
                detail = getattr(e, "stderr", "") or ""
                log(f"match {started}: could not start: {e} {detail}".strip())
                consecutive_failures += 1
                break
            running += 1

        if args.dry_run:
            return 0
        if running == 0 and (stop or (args.matches and started >= args.matches)):
            log(f"all done: {finished} ok, {failed} failed")
            return 0 if failed == 0 else 1
        time.sleep(args.poll)


def main(argv: list[str] | None = None) -> int:
    load_env()
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", default=env("SELFPLAY_IMAGE"), help="client image, e.g. cnn-client:<tag>")
    p.add_argument(
        "--opponent-image",
        default=env("SELFPLAY_OPPONENT_IMAGE"),
        help="evaluation: play --image against this one, alternating sides; default: self-play",
    )
    p.add_argument(
        "--pull-policy", default="IfNotPresent", help="IfNotPresent works for images imported into containerd"
    )
    p.add_argument("--parallel", type=int, default=2, help="matches running at the same time (default 2)")
    p.add_argument("--gpu-per-client", type=int, default=2, help="nvidia.com/gpu slots per client (default 2)")
    p.add_argument("--matches", type=int, default=0, help="total matches to play; 0 = until stopped (default)")
    p.add_argument("--namespace", default=env("SELFPLAY_NAMESPACE"))
    p.add_argument("--run-label", default="default", help="label value tying Jobs to this loop")
    p.add_argument("--server-url", default=env("SELFPLAY_SERVER_URL"), help="server URL as seen from the pods")
    p.add_argument("--client-host", default=env("SELFPLAY_CLIENT_HOST"), help="server URL for the clients if different")
    p.add_argument("--creator-secret", default=env("SELFPLAY_CREATOR_SECRET"), help="default: --team0-secret")
    p.add_argument("--curl-image", default="curlimages/curl:8.17.0", help="image of the match-creating Job")
    p.add_argument("--team0-secret", default=env("SELFPLAY_TEAM0_SECRET"))
    p.add_argument("--team1-secret", default=env("SELFPLAY_TEAM1_SECRET"))
    p.add_argument(
        "--ca-configmap", default="", help="ConfigMap with key ca.crt, for an https server with a private CA"
    )
    p.add_argument("--runtime-class", default=env("SELFPLAY_RUNTIME_CLASS", "nvidia"), help="'' to omit")
    p.add_argument("--image-pull-secret", default=env("SELFPLAY_IMAGE_PULL_SECRET"))
    p.add_argument(
        "--client-env",
        default=env("SELFPLAY_CLIENT_ENV"),
        help="extra environment of the clients, KEY=VALUE[,KEY=VALUE...], e.g. DC_TEAM_NAME_FULL=1",
    )
    p.add_argument("--threads", type=int, default=0, help="search threads per client; 0 = all node CPUs")
    p.add_argument("--deadline", type=int, default=3600, help="activeDeadlineSeconds per client Job")
    p.add_argument("--poll", type=float, default=10, help="seconds between checks")
    p.add_argument("--max-consecutive-failures", type=int, default=3)
    p.add_argument(
        "--match-prefix",
        default=None,
        help="match names: <prefix>-<version(s)>-<time>-m<n> (default: selfplay or eval)",
    )
    g = p.add_argument_group("game")
    # Defaults: the official 10-end setting (DigitalCurling3 wiki, server config v1.1 / GAT2023
    # regulation): 10 ends, 219 s per team, 21.9 s per extra end, fcv1, five-rock (FGZ) rule.
    g.add_argument("--game-mode", default="standard")
    g.add_argument("--tournament", default=None, help="default: selfplay-<version> or eval-<candidate>-vs-<opponent>")
    g.add_argument("--simulator", default="fcv1")
    g.add_argument("--applied-rule", default="fgz_rule")
    g.add_argument("--time-limit", type=float, default=219, help="seconds per team for the standard ends")
    g.add_argument("--extra-end-time-limit", type=float, default=21.9, help="seconds per team per extra end")
    g.add_argument("--end-count", type=int, default=10)
    g.add_argument("--positioned-pattern", type=int, default=0)
    p.add_argument("--verbose", action="store_true", help="also log every match start and finish")
    p.add_argument("--dry-run", action="store_true", help="print one match's Job manifests and exit")
    args = p.parse_args(argv)

    required = {
        "image": "SELFPLAY_IMAGE",
        "namespace": "SELFPLAY_NAMESPACE",
        "server_url": "SELFPLAY_SERVER_URL",
        "team0_secret": "SELFPLAY_TEAM0_SECRET",
        "team1_secret": "SELFPLAY_TEAM1_SECRET",
    }
    missing = [f"--{k.replace('_', '-')} ({v})" for k, v in required.items() if not getattr(args, k)]
    if missing:
        p.error("missing settings, pass them or set them in .env: " + ", ".join(missing))
    args.creator_secret = args.creator_secret or args.team0_secret
    args.match_prefix = args.match_prefix or ("eval" if args.opponent_image else "selfplay")
    args.tournament = args.tournament or default_tournament(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

"""Train the policy/score network on self-play samples (dc4learning.data build).

    python -m dc4learning.train --data data/datasets/<dataset>.npz --init data/models/best.pt --name gen1
    tensorboard --logdir runs        # watch it, http://localhost:6006

Defaults follow the earlier DC3 training (digitalcurling-deeplearning): SGD lr 0.003, momentum 0.9,
weight decay 3e-4, batch 1024, fp16 autocast, loss = policy CE * --policy-weight + score CE.

Train/validation split is by match, so positions of one game never land on both sides.
Every epoch writes to runs/<name>/: TensorBoard scalars, `checkpoint-NNN.pth` (to resume or
compare), and a model the C++ client can load as-is, named after its lineage (dc4learning.naming):
`<gen>-e<epoch>-p<parent hash>-d<dataset hash>-<hash>.pt`, with a sidecar .json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import runs_dir
from .encoding import LEGAL_MASK, encode_batch, mirror_actions, mirror_pixels
from .model import PolicyValueNetwork, export, load
from .naming import HASH, finalize, next_gen, parse, sha256


def parent_model(init: Path | None) -> dict:
    """{"file", "sha256"} of the model training starts from: a traced.pt (through any link), or the
    export a checkpoint was saved with; {None, None} from scratch."""
    if init is None:
        return {"file": None, "sha256": None}
    if init.suffix == ".pth":
        ckpt = torch.load(init, map_location="cpu")
        return {"file": ckpt.get("model_file"), "sha256": ckpt.get("model_sha256")}
    real = init.resolve()
    return {"file": real.name, "sha256": sha256(real)}


def load_dataset(path: Path, device) -> dict[str, torch.Tensor]:
    d = np.load(path)
    keys = ("pixels", "shot", "policy", "score", "match")
    out = {k: torch.from_numpy(d[k].astype(np.int64) if k != "pixels" else d[k]).to(device) for k in keys}
    out["legal"] = torch.from_numpy(LEGAL_MASK).to(device)
    illegal = (out["policy"] >= 0) & ~out["legal"][out["policy"].clamp(min=0)]
    if illegal.any():
        print(
            f"warning: {int(illegal.sum())} played shots are outside the legal mask; dropped from the policy loss",
            file=sys.stderr,
        )
        out["policy"][illegal] = -1
    return out


def split_by_match(
    match: torch.Tensor, val_frac: float, seed: int, train_matches: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """(train, validation) sample indices; whole matches go to one side.

    The validation matches depend only on the seed and val_frac, so runs with different
    `train_matches` (a subset of the remaining matches) are compared on the same games.
    """
    ids = match.unique()
    g = torch.Generator().manual_seed(seed)
    perm = ids[torch.randperm(len(ids), generator=g).to(ids.device)]
    n_val = max(1, round(len(ids) * val_frac)) if val_frac > 0 else 0
    val_ids, train_ids = perm[:n_val], perm[n_val:]
    if train_matches:
        train_ids = train_ids[:train_matches]
    return torch.isin(match, train_ids).nonzero().squeeze(1), torch.isin(match, val_ids).nonzero().squeeze(1)


def losses(model, data, idx, policy_weight: float, amp: bool):
    x = encode_batch(data["pixels"][idx], data["shot"][idx])
    policy_t, score_t = data["policy"][idx], data["score"][idx]
    with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp):
        policy, score = model(x)
    # Like the client, score only legal shots.
    policy = policy.float().masked_fill(~data["legal"], torch.finfo(torch.float32).min)
    score = score.float()
    has_policy = policy_t >= 0
    loss_policy = F.cross_entropy(policy[has_policy], policy_t[has_policy]) if has_policy.any() else policy.sum() * 0
    loss_score = F.cross_entropy(score, score_t)
    return loss_policy * policy_weight + loss_score, loss_policy, loss_score, policy, score, has_policy


@torch.no_grad()
def evaluate(model, data, idx, batch_size: int, policy_weight: float, amp: bool) -> dict[str, float]:
    model.eval()
    sums = {
        k: 0.0 for k in ("loss", "loss_policy", "loss_score", "policy_top1", "policy_top5", "score_acc", "score_mae")
    }
    n = n_policy = 0
    for i in range(0, len(idx), batch_size):
        b = idx[i : i + batch_size]
        loss, lp, ls, policy, score, has_policy = losses(model, data, b, policy_weight, amp)
        k = len(b)
        kp = int(has_policy.sum())
        sums["loss"] += loss.item() * k
        sums["loss_policy"] += lp.item() * kp
        sums["loss_score"] += ls.item() * k
        pt = data["policy"][b][has_policy]
        top5 = policy[has_policy].topk(5, dim=1).indices
        sums["policy_top1"] += (top5[:, 0] == pt).sum().item()
        sums["policy_top5"] += (top5 == pt[:, None]).any(dim=1).sum().item()
        st = data["score"][b]
        sums["score_acc"] += (score.argmax(1) == st).sum().item()
        expected = (score.softmax(1) * torch.arange(score.shape[1], device=score.device)).sum(1)
        sums["score_mae"] += (expected - st).abs().sum().item()
        n += k
        n_policy += kp
    for key in ("loss_policy", "policy_top1", "policy_top5"):
        sums[key] /= max(1, n_policy)
    for key in ("loss", "loss_score", "score_acc", "score_mae"):
        sums[key] /= max(1, n)
    return sums


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="dataset .npz from `dc4learning.data build`")
    p.add_argument("--init", type=Path, help="start from a traced.pt or checkpoint-*.pth; default: random weights")
    p.add_argument("--name", default=None, help="run name (default: timestamp)")
    p.add_argument(
        "--gen",
        default=None,
        help="generation of the exports (default: the parent's + 1 from a model, the same from a checkpoint)",
    )
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--mirror", action="store_true", help="add a left-right mirrored copy of every training sample")
    p.add_argument("--policy-weight", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.05, help="fraction of matches held out")
    p.add_argument("--seed", type=int, default=0, help="also picks the validation matches")
    p.add_argument("--train-matches", type=int, default=0, help="train on only this many matches (0 = all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--log-every", type=int, default=50, help="steps between training-loss points")
    args = p.parse_args(argv)

    from torch.utils.tensorboard import SummaryWriter

    device = torch.device(args.device)
    amp = not args.no_amp and device.type == "cuda"
    name = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = runs_dir() / name
    out.mkdir(parents=True, exist_ok=True)
    # Lineage (dc4learning.naming): every export is named after its parent model and dataset.
    parent = parent_model(args.init)
    data_file = args.data.resolve()
    data_hash = (parse(data_file.name) or {}).get("hash")
    if not data_hash:
        data_hash = sha256(data_file)[:HASH]
        print(
            f"warning: {data_file.name} is not a versioned dataset name; exports point to d{data_hash}", file=sys.stderr
        )
    data_meta_path = data_file.with_suffix(".json")
    data_meta = json.loads(data_meta_path.read_text()) if data_meta_path.exists() else {}
    # A new generation starts from an exported model; resuming a run's checkpoint keeps its generation.
    if args.gen:
        gen = args.gen
    elif args.init and args.init.suffix == ".pth" and parse(parent["file"] or ""):
        gen = parse(parent["file"])["gen"]
    else:
        gen = next_gen(parent["file"])
    run_info = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "gen": gen,
        "parent": parent,
        "dataset": {
            "file": data_file.name,
            "hash": data_hash,
            **{k: data_meta.get(k) for k in ("samples", "played_by")},
        },
    }
    (out / "args.json").write_text(json.dumps(run_info, indent=1))
    params = {k: v for k, v in vars(args).items() if k not in ("data", "init", "name", "device", "log_every")}

    data = load_dataset(args.data, device)
    train_idx, val_idx = split_by_match(data["match"], args.val_frac, args.seed, args.train_matches)
    if args.mirror:
        # Append a left-right mirrored copy of every training sample (validation stays as played).
        n = len(data["shot"])
        tr = train_idx.cpu().numpy()
        pix = torch.from_numpy(mirror_pixels(data["pixels"][train_idx].cpu().numpy())).to(device)
        pol = torch.from_numpy(mirror_actions(data["policy"][train_idx].cpu().numpy())).to(device)
        data["pixels"] = torch.cat([data["pixels"], pix])
        data["policy"] = torch.cat([data["policy"], pol])
        for k in ("shot", "score", "match"):
            data[k] = torch.cat([data[k], data[k][train_idx]])
        train_idx = torch.cat([train_idx, torch.arange(n, n + len(tr), device=device)])
        kept = int((pol >= 0).sum()) / max(1, int((data["policy"][train_idx[: len(tr)]] >= 0).sum()))
        print(f"mirror: {len(tr)} mirrored samples added ({kept:.1%} keep a policy target)", file=sys.stderr)
    print(f"{len(train_idx)} train / {len(val_idx)} validation samples", file=sys.stderr)

    model = (load(args.init) if args.init else PolicyValueNetwork()).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # cosine: decay the learning rate to 0 over all steps (stepped per batch); constant: the DC3 recipe.
    steps_per_epoch = (len(train_idx) + args.batch_size - 1) // args.batch_size
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * steps_per_epoch))
        if args.lr_schedule == "cosine"
        else None
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    writer = SummaryWriter(out)
    writer.add_text("args", json.dumps(vars(args), default=str, indent=1))

    def log_eval(epoch: int, step: int) -> dict[str, float]:
        metrics = evaluate(model, data, val_idx, args.batch_size, args.policy_weight, amp) if len(val_idx) else {}
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, step)
        if metrics:
            print(
                f"epoch {epoch:3d} step {step:6d}  val loss {metrics['loss']:.3f}"
                f"  policy {metrics['loss_policy']:.3f} top1 {metrics['policy_top1']:.3f}"
                f" top5 {metrics['policy_top5']:.3f}  score {metrics['loss_score']:.3f}"
                f" acc {metrics['score_acc']:.3f} mae {metrics['score_mae']:.2f}",
                file=sys.stderr,
            )
        return metrics

    step = 0
    log_eval(0, step)  # baseline: the initial model on the held-out matches
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.monotonic()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        run = {"loss": 0.0, "loss_policy": 0.0, "loss_score": 0.0, "n": 0}
        for i in range(0, len(perm), args.batch_size):
            b = perm[i : i + args.batch_size]
            if len(b) < 2:
                continue  # batch norm needs more than one sample
            loss, lp, ls, *_ = losses(model, data, b, args.policy_weight, amp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched:
                sched.step()
            step += 1
            run["loss"] += loss.item()
            run["loss_policy"] += lp.item()
            run["loss_score"] += ls.item()
            run["n"] += 1
            if step % args.log_every == 0:
                for k in ("loss", "loss_policy", "loss_score"):
                    writer.add_scalar(f"train/{k}", run[k] / run["n"], step)
                run = {"loss": 0.0, "loss_policy": 0.0, "loss_score": 0.0, "n": 0}
        writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
        writer.add_scalar("train/epoch_seconds", time.monotonic() - t0, step)
        metrics = log_eval(epoch, step)
        tmp = out / f".epoch-{epoch:03d}.tmp.pt"
        export(model, tmp)
        exported = finalize(
            tmp,
            f"{gen}-e{epoch:03d}-p{(parent['sha256'] or '0' * HASH)[:HASH]}-d{data_hash}",
            ".pt",
            {
                "kind": "model",
                "gen": gen,
                "epoch": epoch,
                "run": name,
                "parent": parent,
                "dataset": run_info["dataset"],
                "params": params,
                "metrics": metrics,
            },
        )
        print(f"exported {exported.name}", file=sys.stderr)
        torch.save(
            {
                "epoch": epoch,
                "step": step,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "metrics": metrics,
                "args": {k: str(v) for k, v in vars(args).items()},
                "model_file": exported.name,
                "model_sha256": sha256(exported),
            },
            out / f"checkpoint-{epoch:03d}.pth",
        )
        writer.flush()
    writer.close()
    print(f"done: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

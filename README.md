# DigitalCurling4-Learning

Log analysis, training and self-play for [DigitalCurling4](https://github.com/digitalcurling/DigitalCurling4-Server).

Related repositories:

| Repository | Role |
|---|---|
| [DigitalCurling4-Server](https://github.com/pon-pokapoka/DigitalCurling4-Server) (fork) | Game server and its deployment |
| [DigitalCurling4-CNN](https://github.com/pon-pokapoka/DigitalCurling4-CNN) | C++ client (libtorch inference) |

## Environment

Development happens in a dev container with CUDA. On the host you only need the NVIDIA driver
(CUDA 13.4 or newer), Docker with GPU support, and the devcontainer CLI (`npm i -g @devcontainers/cli`)
or VS Code.

```bash
.devcontainer/dev.sh up       # build the image and start the container (runs `uv sync`)
.devcontainer/dev.sh shell    # shell inside it
.devcontainer/dev.sh sync     # `uv sync` again after changing dependencies
.devcontainer/dev.sh down     # stop and remove the container (the venv volume stays)
```

Or open the folder in VS Code and choose "Reopen in Container". Notebooks run through the VS Code Jupyter
extension with the `/opt/venv` kernel.

Personal options (SSH agent, Claude Code, GitHub CLI, git config, private CA for
your server) go in `.devcontainer/local.env`, which is not committed: see
`.devcontainer/local.env.example`. Same options as DigitalCurling4-CNN, so you can copy that file.

| | Version | Why |
|---|---|---|
| CUDA base image | `nvidia/cuda:13.4.3-base-ubuntu26.04` | CUDA libraries come from the PyTorch wheels; only the driver is needed |
| PyTorch | `2.15.0+cu134` | Must match the libtorch in DigitalCurling4-CNN so exported models load there. Bump both together |
| Python | 3.13 | `torch.jit.trace` (TorchScript) is unsupported on 3.14+ |

Dependencies are managed with uv; `uv.lock` is committed. The venv is at `/opt/venv` on the
`dc4-learning-venv` volume and uv's cache on `dc4-learning-uv-cache`, so rebuilding the container
doesn't download torch again.

### Cluster image

The same `Dockerfile` has a `train` target with the code and locked dependencies baked in (no dev group,
non-root):

```bash
docker build --target train -t ghcr.io/pon-pokapoka/dc4-learning:<tag> .
```

## Layout

- `src/dc4learning/api.py`: read-only client for the server's REST API (GET endpoints, no auth, retries on 5xx).
  Server URL comes from `DC_SERVER_URL`, from the environment or the repo's `.env`.
- `src/dc4learning/sheet.py`: sheet geometry (meters), drawing helpers, and the DC3-style pixel grid.
- `src/dc4learning/selfplay.py`: self-play and evaluation matches on the cluster.
- `src/dc4learning/data.py`: reads finished matches from the server and builds training datasets.
- `src/dc4learning/encoding.py`: model input and targets, exactly as the C++ client builds and reads them.
- `src/dc4learning/model.py`: the client's network as a trainable module; loads and exports `traced.pt`.
- `src/dc4learning/train.py`: training, with TensorBoard logs and client-ready exports every epoch.
- `src/dc4learning/stats.py`: self-play status and results.
- `src/dc4learning/versions.py`: which client and model played each match, and each model's lineage.
- `src/dc4learning/naming.py`: file names and sidecars that record where models, datasets and tables came from.
- `src/dc4learning/wintable.py`: the client's win table, estimated from self-play games.
- `scripts/client-image.sh`, `scripts/model-image.sh`: client images from a CNN commit or with a trained
  model, recorded and loaded into the cluster.
- `notebooks/inspect_latest_game.ipynb`: draws every shot of the latest match.
- `notebooks/selfplay_analysis.ipynb`: throughput, results, points per end, win table, shots played, model fit.

## Self-play on the cluster

`python -m dc4learning.selfplay` keeps `--parallel` matches (default 2) running in a Kubernetes namespace
and starts a new one as soon as one finishes. Each match is one server match plus two Jobs, one CNN
client per team, each requesting `--gpu-per-client` time-sliced `nvidia.com/gpu` slots (default 2, so
2 matches use 8 slots). Jobs are labelled with their match ID, so the loop can be restarted without
losing running matches. Ctrl-C stops starting new matches; a second Ctrl-C quits.

```bash
cp .env.example .env   # fill in server URL, image, namespace, Secret names
python -m dc4learning.selfplay              # until stopped
python -m dc4learning.selfplay -h           # all options
python -m dc4learning.selfplay --dry-run    # print the Jobs
```

Cluster-specific settings have no defaults; they come from options, environment variables, or the
uncommitted `.env`. No credentials pass through the machine running the loop: each match is
created by a short Job with `curlimages/curl` that logs in with a Secret, and the loop itself only needs
kubectl access to the namespace. Game defaults are the official 10-end setting (DigitalCurling3 wiki:
219 s per team, 21.9 s per extra end, fcv1, five-rock rule).

The namespace needs one Secret per team (keys `auth-id`, `auth-pw`) and the client image on the GPU
node (`imagePullPolicy: IfNotPresent`, so an image imported into containerd works). Pods use
`runtimeClassName: nvidia` by default; without the NVIDIA runtime they get a GPU slot but no device.

## Logs, training and evaluation

The loop is: self-play with the current client, read the games from the server, train, build a client image with
the trained model, play it against the current one, and promote it if it wins.

```bash
scripts/client-image.sh ../DigitalCurling4-CNN origin/development gen0   # cnn-client:<commit>, model named gen0
python -m dc4learning.data build --model gen0          # data/datasets/selfplay-gen0-n<matches>-<hash>.npz
python -m dc4learning.train --data data/datasets/selfplay-gen0-n<...>.npz --init data/models/best.pt --name gen1
scripts/model-image.sh runs/gen1/gen1-e007-p<...>.pt   # cnn-client:<commit>_gen1-<hash>, imported via IMAGE_IMPORT_CMD
python -m dc4learning.selfplay --opponent-image cnn-client:<commit>_gen1-<hash> --matches 200   # evaluation
python -m dc4learning.wintable --model gen0            # optional: a win table from the same games
python -m dc4learning.stats                            # wins per client/model pair, binomial test
```

Training targets come from the logs: the policy target is the shot the client chose (the server's
`actual_*` fields, before execution noise; note that the server logs the negative of the client's
angular velocity), the score target is the hammer team's net points in that end. Validation holds
out whole matches. `--init` takes the client's `traced.pt` or a checkpoint; a traced file can't be
fine-tuned directly because its batch norms were traced in eval mode, so `model.py` rebuilds the
network and copies the weights.

Match names record the client images (`selfplay-<image>-...`, `selfplay-<image0>-vs-<image1>-...`),
which is how `data` and `stats` tell generations apart. In evaluation the two images alternate sides.

### Versions

Every match is attributable to a client commit and a model, and every trained model to its parent
model and training games (`python -m dc4learning.versions list`, `... show <match_id>`):

- An image is identified by its Docker image ID (`data/versions/images.json`); a match by the version ID
  in its server-side match and tournament names. Matches played before the names carried version IDs
  are linked through per-match records in `data/versions/matches/` (no longer written for new matches).
- A model is identified by the sha256 of its `traced.pt`; the CNN repo doesn't track model files, so its
  commit alone doesn't say which model an image runs. Names (`gen0`, `gen1`, ...) are labels on top.
- File names record where a file came from (`dc4learning/naming.py`). Every artifact ends in the first
  8 hex digits of its own sha256 and names its inputs by theirs, so the chain can be followed by file
  name alone; a sidecar `<same name>.json` adds the commit of this repo that made it, the parameters,
  and the match IDs:

  | Artifact | Name | Example |
  |---|---|---|
  | model | `<gen>-e<epoch>-p<parent>-d<dataset>-<hash>.pt` | `gen1-e007-p588f425c-d372b3689-3f2a9c1d.pt` |
  | dataset | `<selection>-n<matches>-<hash>.npz` | `selfplay-gen0-n539-372b3689.npz` |
  | win table | `win_table-<selection>-n<matches>-<hash>.csv` | `win_table-selfplay-gen0-n539-ef24445e.csv` |
  | root (origin not recorded) | `<gen>-<hash>.pt`, `win_table-<hash>.csv` | `gen0-588f425c.pt` |

  `train` names its exports this way (the generation is the parent's + 1, or the same when resuming a
  checkpoint), `data build` its datasets and `wintable` its tables. `python -m dc4learning.versions trace
  <file>` follows the chain and checks each file still matches its hash.
- Consumers use links with the names they expect: `src/engine/model/traced.pt` and `win_table.csv` in the
  CNN repo, `data/models/best.pt` here for the current champion. `model-image.sh` keeps each model it
  packages (with its sidecar) in `data/models/` and tags the image `<base tag>_<gen>-<hash>`; promoting a
  generation is repointing `best.pt` (and, for new client builds, the CNN repo's links).
- `client-image.sh` and `model-image.sh` record images and models and also stamp the facts into the
  image as labels; `train` writes the starting model's sha256 and the dataset's makeup into
  `runs/<name>/args.json`, which `model-image.sh` reads for the lineage.
- `data` and `stats` add `client0/1`, `model0/1` and `image_id0/1` to every match; `data build` can
  select by `--model` or `--client` and writes `<dataset>.json` with the versions that played its games.

Nothing the server's database already holds is kept here: games are read from the server when a
dataset, win table or report needs them (`data.games`, `data.load_match`). `data/` (not committed)
keeps only what the server doesn't have, so back it up: model files and sidecars, datasets, win
tables, the image/model catalog, and the per-match records of older matches (see above). Match and
tournament names carry each team's version ID
(`<client commit>.<model>.<model hash>`), so the server's records say what played each game.

### Monitoring

| What | How |
|---|---|
| The self-play loop | its log: `tail -f logs/selfplay.log` (if started with `>> logs/selfplay.log`); only failures, warnings and stops unless `--verbose` |
| Matches on the cluster | `python -m dc4learning.stats` (reads the games from the server), or `kubectl -n <namespace> get jobs` |
| A running match | the server's viewer, `<server>/matches/<match_id>/viewer`; or `kubectl -n <namespace> logs job/<job>` |
| Results so far | `watch -n 300 python -m dc4learning.stats`; `notebooks/selfplay_analysis.ipynb` for plots |
| Training | `tensorboard --logdir runs` (port 6006): train/val losses, policy top-1/top-5, score accuracy |

To keep self-play running after you log out, start it detached:
`setsid nohup python -m dc4learning.selfplay >> logs/selfplay.log 2>&1 &`. Training shares the GPU with
the self-play clients; it fits (about 1.5 GB), but both slow down while it runs.

## Coordinates

Simulator coordinates in meters: `x` across the sheet (0 = center line), `y` along the sheet.
Tee line `y = 38.405`, hog line `y = 32.004`, house radius 1.829, stone radius 0.145.
A stone at `(0, 0)` is not in play.

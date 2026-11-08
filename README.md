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

- `src/dc4learning/api.py`: read-only client for the server's REST API (GET endpoints, no auth).
  Server URL comes from `DC_SERVER_URL`, from the environment or the repo's `.env`.
- `src/dc4learning/sheet.py`: sheet geometry (meters), drawing helpers, and the DC3-style pixel grid used as training input.
- `src/dc4learning/selfplay.py`: self-play on the cluster (see below).
- `notebooks/inspect_latest_game.ipynb`: draws every shot of the latest match.

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

## Coordinates

Simulator coordinates in meters: `x` across the sheet (0 = center line), `y` along the sheet.
Tee line `y = 38.405`, hog line `y = 32.004`, house radius 1.829, stone radius 0.145.
A stone at `(0, 0)` is not in play.

# syntax=docker/dockerfile:1
#
# One image definition, two targets:
#   dev   : the dev container (.devcontainer/devcontainer.json). The source tree is
#           bind-mounted; the venv lives in a named volume and `uv sync` fills it.
#   train : for cluster Jobs. Code + locked dependencies baked in, non-root.
#           docker build --target train -t ghcr.io/pon-pokapoka/dc4-learning:<tag> .
#
# CUDA comes from the PyTorch wheels (nvidia-* packages), so the *base* CUDA image
# is enough; only the host driver is needed. Switch to -cudnn-devel only if you
# need nvcc to compile custom CUDA extensions.
#
# Keep in sync with DigitalCurling4-CNN (libtorch 2.15.0+cu134, CUDA 13.4.3,
# Ubuntu 26.04) so exported models load in the C++ client. torch itself is pinned
# in pyproject.toml.

ARG CUDA_VERSION=13.4.3
ARG UBUNTU_VERSION=26.04
ARG UV_VERSION=0.13.7

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION} AS base

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# Python is installed by uv into the image (not the venv volume). Keep
# PYTHON_VERSION equal to .python-version. Stay on 3.13 for now: torch.jit.trace
# (TorchScript, what the C++ client loads) is unsupported on Python 3.14+.
# Downloads are disabled afterwards so a mismatch fails loudly instead of
# installing a Python that a container recreate would lose.
ARG PYTHON_VERSION=3.13
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN uv python install ${PYTHON_VERSION}
ENV UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:${PATH}

# ---------------------------------------------------------------------------
FROM base AS train

WORKDIR /app
# Dependencies first (cached unless the lock changes), then the package.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-dev --no-install-project
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

RUN useradd --create-home --uid 10001 app
USER app

# ---------------------------------------------------------------------------
FROM base AS dev

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bash-completion \
        curl \
        gh \
        git \
        jq \
        less \
        openssh-client \
        sudo \
        vim \
    && rm -rf /var/lib/apt/lists/*

# So git over SSH to GitHub works without a prompt in a fresh container.
COPY .devcontainer/known_hosts /etc/ssh/ssh_known_hosts

# personal git config (GIT_CONFIG_GLOBAL), backed by a named volume when
# DC4_GIT_CONFIG=1 (see dev.sh)
RUN mkdir -p /opt/git-config

# Claude Code (native installer), only with INSTALL_CLAUDE_CODE=1 (off by default;
# see .devcontainer/local.env.example). Rebuild the image to update.
ARG INSTALL_CLAUDE_CODE=0
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN if [ "${INSTALL_CLAUDE_CODE}" = "1" ]; then \
        curl -fsSL https://claude.ai/install.sh | bash \
        && cp -L /root/.local/bin/claude /usr/local/bin/claude \
        && rm -rf /root/.local/bin/claude /root/.local/share/claude; \
    fi \
    && mkdir -p /opt/claude-config
ENV CLAUDE_CONFIG_DIR=/opt/claude-config \
    DISABLE_AUTOUPDATER=1

#!/usr/bin/env bash
# Thin wrapper around the devcontainer CLI (npm i -g @devcontainers/cli).
#
#   .devcontainer/dev.sh up        # build image (if needed) and start the container
#   .devcontainer/dev.sh shell     # interactive shell inside the container
#   .devcontainer/dev.sh sync      # uv sync inside the container
#   .devcontainer/dev.sh exec CMD  # run an arbitrary command inside the container
#   .devcontainer/dev.sh rebuild   # rebuild image from scratch and recreate container
#   .devcontainer/dev.sh down      # stop and remove the container
#
# Personal options (SSH agent, Claude Code, GitHub CLI, git config, private CA) are
# read from .devcontainer/local.env, which is not committed; see
# local.env.example. Without it you get a plain container that needs no SSH key,
# GitHub or Claude and creates no personal volumes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="dc4-learning-dev"   # keep in sync with runArgs in devcontainer.json
AGENT_TARGET="/run/host-ssh-agent.sock"
CA_TARGET="/run/dc4-ca.crt"

LOCAL_ENV="$REPO_ROOT/.devcontainer/local.env"
if [[ -f "$LOCAL_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$LOCAL_ENV"
fi
# Read by devcontainer.json (${localEnv:...}) as an image build argument.
export DC4_CLAUDE_CODE="${DC4_CLAUDE_CODE:-0}"
DC4_SSH_AGENT="${DC4_SSH_AGENT:-0}"
DC4_GITHUB="${DC4_GITHUB:-0}"
DC4_GIT_CONFIG="${DC4_GIT_CONFIG:-0}"

if command -v devcontainer >/dev/null 2>&1; then
  DC=(devcontainer)
elif command -v npx >/dev/null 2>&1; then
  DC=(npx --yes @devcontainers/cli)
else
  echo "devcontainer CLI not found. Install with: npm install -g @devcontainers/cli" >&2
  exit 1
fi

# Mounts only apply when the container is created: after changing an option,
# run down + up (or rebuild for DC4_CLAUDE_CODE).
up_args=()
exec_args=()
# add_volume <volume> <target> [ENV=value]: a named volume, created by Docker on
# first use, plus an environment variable pointing at it.
add_volume() {
  up_args+=(--mount "type=volume,source=$1,target=$2")
  if [[ -n "${3:-}" ]]; then
    up_args+=(--remote-env "$3")
    exec_args+=(--remote-env "$3")
  fi
}
# Claude Code login and settings (CLAUDE_CONFIG_DIR is set in the image).
[[ "$DC4_CLAUDE_CODE" == 1 ]] && add_volume dc4-learning-claude /opt/claude-config
# GitHub CLI login.
[[ "$DC4_GITHUB" == 1 ]] && add_volume dc4-learning-gh /opt/gh-config GH_CONFIG_DIR=/opt/gh-config
# Personal git config (identity, signing setup), kept out of the repo.
[[ "$DC4_GIT_CONFIG" == 1 ]] && add_volume dc4-learning-git /opt/git-config GIT_CONFIG_GLOBAL=/opt/git-config/gitconfig

# DC4_SSH_AGENT=1: bind-mount the host ssh-agent socket (for git over SSH and
# commit signing). The private key never enters the container.
if [[ "$DC4_SSH_AGENT" == 1 ]]; then
  if [[ -z "${SSH_AUTH_SOCK:-}" || ! -S "$SSH_AUTH_SOCK" ]]; then
    echo "DC4_SSH_AGENT=1 but there is no ssh-agent on the host (SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-unset})." >&2
    echo "Start one and load your key first:  eval \"\$(ssh-agent)\" && ssh-add" >&2
    exit 1
  fi
  ssh-add -l >/dev/null 2>&1 || echo "Warning: the host ssh-agent has no keys loaded; run ssh-add." >&2
  up_args+=(--mount "type=bind,source=$SSH_AUTH_SOCK,target=$AGENT_TARGET"
            --remote-env "SSH_AUTH_SOCK=$AGENT_TARGET")
  exec_args+=(--remote-env "SSH_AUTH_SOCK=$AGENT_TARGET")
fi

# DC4_CA_CERT=<host path>: mount that public certificate; setup-ca.sh trusts it
# when the container starts.
if [[ -n "${DC4_CA_CERT:-}" ]]; then
  ca_file="$(readlink -f "$DC4_CA_CERT" || true)"
  if [[ ! -f "$ca_file" ]]; then
    echo "DC4_CA_CERT=$DC4_CA_CERT is not a file on the host." >&2
    exit 1
  fi
  up_args+=(--mount "type=bind,source=$ca_file,target=$CA_TARGET"
            --remote-env "DC4_CA_CERT=$CA_TARGET")
  exec_args+=(--remote-env "DC4_CA_CERT=$CA_TARGET")
fi

dc_up() { "${DC[@]}" up --workspace-folder "$REPO_ROOT" "${up_args[@]}" "$@"; }
dc_exec() { "${DC[@]}" exec --workspace-folder "$REPO_ROOT" "${exec_args[@]}" "$@"; }

cmd="${1:-shell}"; shift || true
case "$cmd" in
  up)      dc_up ;;
  shell)   dc_up >/dev/null && dc_exec bash -l ;;
  sync)    dc_up >/dev/null && dc_exec uv sync "$@" ;;
  exec)    dc_up >/dev/null && dc_exec "$@" ;;
  rebuild) dc_up --remove-existing-container --build-no-cache ;;
  down)    docker rm -f "$CONTAINER_NAME" ;;
  *)       sed -n '2,14p' "$0"; exit 1 ;;
esac

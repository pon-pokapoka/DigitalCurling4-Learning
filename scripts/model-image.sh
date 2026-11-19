#!/usr/bin/env bash
# Build a CNN client image that runs a trained model, record it, and optionally load it into the cluster.
#
#   scripts/model-image.sh <model.pt> [<model name>] [<tag>]
#   scripts/model-image.sh runs/gen1/gen1-e007-p588f425c-d1a2b3c4d-3f2a9c1d.pt
#
# The image is the client image in BASE_IMAGE (default SELFPLAY_IMAGE) with its model replaced,
# tagged <name of the base image>:<tag>, default <base tag>_<gen>-<model hash>
# (e.g. cnn-client:a604b03_gen1-3f2a9c1d). Same client binary and win table, only traced.pt differs.
# The base may be in a registry (e.g. registry.example/cnn-client:v26.11.0); it is pulled if needed.
# The model name defaults to the generation in the file name (dc4learning.naming).
#
# The model file and its sidecar .json are kept in data/models under their versioned name (a model
# without one is kept as <name>-<hash>.pt); point data/models/best.pt at it to promote it.
# Recorded in data/versions (dc4learning.versions): the model (sha256, name, and for a training
# export its run, starting model and training games) and the image. The image also carries these as
# labels, so they travel with it.
#
# If IMAGE_REGISTRY is set (environment or the repo's .env), the image is pushed there as
# <registry>/<name>:<tag> (docker login first). Otherwise, if IMAGE_IMPORT_CMD is set, it is piped
# into it as a `docker save` tarball, e.g. IMAGE_IMPORT_CMD='sudo ctr -n k8s.io images import -'.
# The last line printed is the reference to give the pods; evaluate it with
# python -m dc4learning.selfplay --opponent-image <that reference> --matches 100
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  sed -n '2,23p' "$0" >&2
  exit 1
fi
model="$(readlink -f "$1")"   # the versioned file behind any link
model_file="$(basename "$model")"
model_name="${2:-}"
if [[ -z "$model_name" && "$model_file" =~ ^(gen[0-9]+)- ]]; then
  model_name="${BASH_REMATCH[1]}"
fi
[[ -n "$model_name" ]] || { echo "$model_file has no generation in its name: pass a model name." >&2; exit 1; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib.sh
source "$repo_root/scripts/lib.sh"

base="${BASE_IMAGE:-${SELFPLAY_IMAGE:-}}"
[[ -n "$base" ]] || { echo "Set BASE_IMAGE or SELFPLAY_IMAGE to the client image to start from." >&2; exit 1; }
[[ -f "$model" ]] || { echo "No model file: $model" >&2; exit 1; }
use_base_image "$base"

# The model, with its lineage if it is a training export.
sha="$(versions register-model "$model" --name "$model_name" | cut -d' ' -f1)"
tag="${3:-${base##*:}_${model_name}-${sha:0:8}}"
image="${base%:*}:$tag"
models_dir="${DC4_DATA_DIR:-$repo_root/data}/models"
mkdir -p "$models_dir"
if [[ "$model_file" == *"-${sha:0:8}.pt" ]]; then
  archived="$models_dir/$model_file"
else
  archived="$models_dir/$model_name-${sha:0:8}.pt"
  echo "warning: $model_file is not a versioned name; keeping it as $(basename "$archived")" >&2
fi
[[ -e "$archived" ]] || cp "$model" "$archived"
sidecar="${model%.pt}.json"
[[ -f "$sidecar" && ! -e "${archived%.pt}.json" ]] && cp "$sidecar" "${archived%.pt}.json"
echo "model $model_name ${sha:0:12} kept as $archived" >&2
client="$(docker image inspect "$base" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' 2>/dev/null || true)"

ctx="$(mktemp -d)"
trap 'rm -rf "$ctx"' EXIT
cp "$model" "$ctx/traced.pt"
cat > "$ctx/Dockerfile" <<EOF
FROM $base
# The client resolves its model as <exe>/../../../src/engine/model (see the CNN Dockerfile).
COPY --chown=10001:10001 traced.pt /app/src/engine/model/traced.pt
EOF
docker build -q -t "$image" \
  --label "io.github.dc4learning.model.sha256=$sha" \
  --label "io.github.dc4learning.model.name=$model_name" \
  --label "io.github.dc4learning.model.source=$model_file" \
  --label "io.github.dc4learning.parent-image=$base" \
  "$ctx" >/dev/null
versions register-image "$image" ${client:+--client-commit "$client"} >&2
publish_image "$image"

#!/usr/bin/env bash
# Build a CNN client image that runs a trained model, record it, and optionally load it into the cluster.
#
#   scripts/model-image.sh <traced.pt> <model name> [<tag>]
#   scripts/model-image.sh runs/gen1/traced-007.pt gen1
#
# The image is the client image in BASE_IMAGE (default SELFPLAY_IMAGE) with its model replaced,
# tagged <name of the base image>:<tag>, default <base tag>_<model name> (e.g. cnn-client:a604b03_gen1).
# Same client binary and win table, only traced.pt differs.
#
# The model is also kept as data/models/<model name>-<sha256 prefix>.pt (the versioned original;
# point data/models/best.pt at it to promote it).
# Recorded in data/versions (dc4learning.versions): the model (sha256, name, and for a training
# export its run, starting model and training games) and the image. The image also carries these as
# labels, so they travel with it.
#
# If IMAGE_IMPORT_CMD is set (environment or the repo's .env), the image is piped into it as a
# `docker save` tarball, e.g. IMAGE_IMPORT_CMD='sudo ctr -n k8s.io images import -'.
# Then evaluate it: python -m dc4learning.selfplay --opponent-image <name>:<tag> --matches 100
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  sed -n '2,18p' "$0" >&2
  exit 1
fi
model="$1"
model_name="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib.sh
source "$repo_root/scripts/lib.sh"

base="${BASE_IMAGE:-${SELFPLAY_IMAGE:-}}"
[[ -n "$base" ]] || { echo "Set BASE_IMAGE or SELFPLAY_IMAGE to the client image to start from." >&2; exit 1; }
[[ -f "$model" ]] || { echo "No model file: $model" >&2; exit 1; }
tag="${3:-${base##*:}_${model_name}}"
image="${base%:*}:$tag"

# The model, with its lineage if it is a training export.
sha="$(versions register-model "$model" --name "$model_name" | cut -d' ' -f1)"
models_dir="${DC4_DATA_DIR:-$repo_root/data}/models"
mkdir -p "$models_dir"
archived="$models_dir/$model_name-${sha:0:8}.pt"
[[ -e "$archived" ]] || cp "$model" "$archived"
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
  --label "io.github.dc4learning.model.source=$model" \
  --label "io.github.dc4learning.parent-image=$base" \
  "$ctx" >/dev/null
versions register-image "$image" ${client:+--client-commit "$client"}
import_image "$image"

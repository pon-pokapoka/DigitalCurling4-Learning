#!/usr/bin/env bash
# Build the CNN client image from a DigitalCurling4-CNN commit, record it, and optionally load it
# into the cluster.
#
#   scripts/client-image.sh <DigitalCurling4-CNN checkout> [<git ref>] [<model name>]
#   scripts/client-image.sh ../DigitalCurling4-CNN origin/development gen0
#
# The image is cnn-client:<short commit> (set IMAGE_NAME to change cnn-client). It is built from
# the committed sources at <git ref> (default HEAD), so local changes don't leak in, plus the model
# files from the checkout's src/engine/model/, which the CNN repo doesn't track. There traced.pt and
# win_table.csv are links to versioned files (e.g. traced.pt -> gen0-588f425c.pt); the model name
# defaults to the part before the hash.
#
# Recorded in data/versions (dc4learning.versions): the client commit, the model's sha256 and name,
# and the image ID. The image carries the same facts as labels.
# If IMAGE_IMPORT_CMD is set (environment or the repo's .env), the image is loaded with it,
# e.g. IMAGE_IMPORT_CMD='sudo ctr -n k8s.io images import -'.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  sed -n '2,16p' "$0" >&2
  exit 1
fi
cnn="$(cd "$1" && pwd)"
ref="${2:-HEAD}"
model_name="${3:-}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib.sh
source "$repo_root/scripts/lib.sh"

commit="$(git -C "$cnn" rev-parse --short "$ref^{commit}")"
model="$cnn/src/engine/model/traced.pt"
[[ -f "$model" && -f "$cnn/src/engine/model/win_table.csv" ]] || {
  echo "Missing $cnn/src/engine/model/{traced.pt,win_table.csv}." >&2; exit 1; }
image="${IMAGE_NAME:-cnn-client}:$commit"
model_file="$(basename "$(readlink -f "$model")")"   # the versioned file behind the link
if [[ -z "$model_name" && "$model_file" =~ ^(.+)-[0-9a-f]{8}\.pt$ ]]; then
  model_name="${BASH_REMATCH[1]}"
fi

ctx="$(mktemp -d)"
trap 'rm -rf "$ctx"' EXIT
git -C "$cnn" archive "$commit" | tar -x -C "$ctx"
mkdir -p "$ctx/src/engine/model"
cp "$model" "$cnn/src/engine/model/win_table.csv" "$ctx/src/engine/model/"

sha="$(versions register-model "$model" ${model_name:+--name "$model_name"} | cut -d' ' -f1)"
echo "building $image from $commit (model ${sha:0:12}); the first build takes a while" >&2
# The build has no .git (git archive), so pass the client its build ID, as `git describe` names
# the commit (e.g. v2026.07-70-gddff1ad). Ignored by clients older than DigitalCurling4-CNN #33.
build_id="$(git -C "$cnn" describe --tags --always "$commit")"
docker build -q -t "$image" \
  --build-arg "BUILD_ID=$build_id" \
  --label "io.github.dc4learning.client.build-id=$build_id" \
  --label "org.opencontainers.image.revision=$commit" \
  --label "io.github.dc4learning.model.sha256=$sha" \
  ${model_name:+--label "io.github.dc4learning.model.name=$model_name"} \
  --label "io.github.dc4learning.model.source=DigitalCurling4-CNN src/engine/model/$model_file" \
  "$ctx" >/dev/null
versions register-image "$image" --client-commit "$commit"
import_image "$image"

# Shared by the image scripts. Sourced with repo_root set.

# KEY=VALUE lines from the repo's .env, without overriding variables already set (even to "").
if [[ -f "$repo_root/.env" ]]; then
  while IFS='=' read -r key value; do
    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
    [[ -v "$key" ]] && continue
    value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
    export "$key=$value"
  done < "$repo_root/.env"
fi

# dc4learning.versions with whatever Python is at hand (it needs only the standard library).
versions() {
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" -m dc4learning.versions "$@"
}

# Reference of an image in IMAGE_REGISTRY: <registry>/<name>:<tag>, keeping a ref that is already there.
registry_ref() {
  local registry="${IMAGE_REGISTRY%/}"
  if [[ "$1" == "$registry/"* ]]; then echo "$1"; else echo "$registry/${1##*/}"; fi
}

# A base image to build on: pulled if it isn't local (e.g. a release pushed to the registry), and
# recorded, so its client commit and model are known (from its labels).
use_base_image() {
  docker image inspect "$1" >/dev/null 2>&1 || docker pull -q "$1" >&2
  versions register-image "$1" >/dev/null
}

# Make an image available where the cluster runs it and print the reference the pods should use:
# pushed to IMAGE_REGISTRY if set (and recorded under that reference too), else loaded with
# IMAGE_IMPORT_CMD.
publish_image() {
  local image="$1" remote
  if [[ -n "${IMAGE_REGISTRY:-}" ]]; then
    remote="$(registry_ref "$image")"
    [[ "$remote" == "$image" ]] || docker tag "$image" "$remote"
    docker push -q "$remote" >&2
    versions register-image "$remote" >/dev/null
    echo "pushed $remote" >&2
    echo "$remote"
    return
  fi
  if [[ -n "${IMAGE_IMPORT_CMD:-}" ]]; then
    docker save "$image" | bash -c "$IMAGE_IMPORT_CMD"
    echo "imported $image" >&2
  else
    echo "Neither IMAGE_REGISTRY nor IMAGE_IMPORT_CMD is set: load $image into the cluster yourself." >&2
  fi
  echo "$image"
}

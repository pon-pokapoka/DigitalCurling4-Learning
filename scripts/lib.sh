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

# Load an image where the cluster runs it, through IMAGE_IMPORT_CMD.
import_image() {
  if [[ -n "${IMAGE_IMPORT_CMD:-}" ]]; then
    docker save "$1" | bash -c "$IMAGE_IMPORT_CMD"
    echo "imported $1" >&2
  else
    echo "IMAGE_IMPORT_CMD is not set: load $1 into the cluster yourself." >&2
  fi
}

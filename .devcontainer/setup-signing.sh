#!/usr/bin/env bash
# Configure SSH commit/tag signing inside the dev container.
#
# The private key stays on the host: the host's ssh-agent socket is bind-mounted
# into the container (see devcontainer.json), and git signs through it. Settings
# are written to GIT_CONFIG_GLOBAL (the dc4-learning-git volume), never to the repo.
# Needs DC4_SSH_AGENT=1 and DC4_GIT_CONFIG=1 in .devcontainer/local.env.
#
#   .devcontainer/setup-signing.sh            # sign with the agent key registered on GitHub
#   .devcontainer/setup-signing.sh <pattern>  # pin the agent key whose line matches <pattern>
#   .devcontainer/setup-signing.sh --check    # refresh the key list and verify (run on container start;
#                                             # silent unless signing is set up)

#
# By default no key is written to the config. The account's GitHub signing keys
# (public API, no login needed) are cached in the volume, and git runs
# gpg.ssh.defaultKeyCommand -> signing-key, which picks the first agent key in
# that cache at each signature. If no agent key is registered, signing fails
# instead of producing commits GitHub shows as "Unverified".
# The GitHub user is $GITHUB_USER, else the gh login, else the owner of origin.
set -euo pipefail

die()  { echo "setup-signing: $*" >&2; exit 1; }
warn() { echo "setup-signing: WARNING: $*" >&2; }

if [[ -z "${GIT_CONFIG_GLOBAL:-}" ]]; then
  # No persistent git config volume: signing is not set up, so --check stays silent.
  [[ "${1:-}" == "--check" ]] && exit 0
  die "GIT_CONFIG_GLOBAL is not set. Set DC4_GIT_CONFIG=1 (and DC4_SSH_AGENT=1) in .devcontainer/local.env, then .devcontainer/dev.sh down && .devcontainer/dev.sh up."
fi
confdir="$(dirname "$GIT_CONFIG_GLOBAL")"
registered="$confdir/github_signing_keys"   # "<type> <blob>" per line
helper="$confdir/signing-key"               # gpg.ssh.defaultKeyCommand
allowed="$confdir/allowed_signers"

agent_keys() {
  [[ -n "${SSH_AUTH_SOCK:-}" && -S "$SSH_AUTH_SOCK" ]] || return 2
  ssh-add -L 2>/dev/null | grep -E '^(ssh-|ecdsa-|sk-)' || return 1
}

github_user() {
  if [[ -n "${GITHUB_USER:-}" ]]; then echo "$GITHUB_USER"; return; fi
  if command -v gh >/dev/null && gh api user --jq .login 2>/dev/null; then return; fi
  git remote get-url origin 2>/dev/null | sed -nE 's#^(git@|ssh://git@|https://)github\.com[:/]([^/]+)/.*#\2#p'
}

# Cache the user's GitHub signing keys. Returns 1 (cache untouched) if offline.
refresh_registered() {
  local user json keys
  user="$(github_user)"
  [[ -n "$user" ]] || { warn "cannot tell the GitHub user; set GITHUB_USER."; return 1; }
  json="$(curl -fsS --max-time 10 "https://api.github.com/users/$user/ssh_signing_keys")" || return 1
  keys="$(python3 -I -c 'import json,sys; [print(*k["key"].split()[:2]) for k in json.load(sys.stdin)]' <<<"$json")" || return 1
  mkdir -p "$confdir"
  printf '%s\n' "$keys" | sed '/^$/d' > "$registered.tmp" && mv "$registered.tmp" "$registered"
}

install_helper() {
  cat > "$helper" <<'EOF'
#!/usr/bin/env bash
# gpg.ssh.defaultKeyCommand: print the first ssh-agent key that is registered on
# GitHub as a signing key (cached by .devcontainer/setup-signing.sh).
registered="$(dirname "$0")/github_signing_keys"
[[ -s "$registered" ]] || { echo "signing-key: no cached GitHub signing keys; run .devcontainer/setup-signing.sh" >&2; exit 1; }
agent="$(ssh-add -L)" || { echo "signing-key: no keys in the forwarded ssh-agent" >&2; exit 1; }
key="$(awk 'NR==FNR { reg[$1" "$2]=1; next } ($1" "$2) in reg { print $1" "$2; exit }' "$registered" - <<<"$agent")"
[[ -n "$key" ]] || { echo "signing-key: no ssh-agent key is registered on GitHub as a signing key; ssh-add it on the host, or register it and rerun .devcontainer/setup-signing.sh" >&2; exit 1; }
echo "key::$key"
EOF
  chmod 755 "$helper"
}

# The key git will sign with ("<type> <blob>"), or nothing.
signing_key() {
  local key
  key="$(git config --global --get user.signingkey || true)"
  if [[ -n "$key" ]]; then echo "${key#key::}"; return; fi
  [[ -x "$helper" ]] && "$helper" 2>/dev/null | sed 's/^key:://'
}

fingerprint() { ssh-keygen -lf <(echo "$1") | awk '{print $2}'; }

check() {
  local key
  # Signing is opt-in: stay silent for anyone who has not set it up.
  [[ "$(git config --global --get commit.gpgsign || true)" == "true" ]] || return 0
  if ! agent_keys >/dev/null; then
    warn "no ssh-agent keys reachable (SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-unset}); commits will fail until the host agent is forwarded. Set DC4_SSH_AGENT=1 in .devcontainer/local.env, 'ssh-add' on the host, then .devcontainer/dev.sh down && .devcontainer/dev.sh up."
    return 0
  fi
  if [[ -z "$(git config --global --get user.signingkey || true)" ]]; then
    refresh_registered || warn "could not refresh GitHub signing keys (offline?); using the cached list."
    install_helper
  fi
  key="$(signing_key || true)"
  if [[ -z "$key" ]]; then
    warn "no key in the forwarded ssh-agent is registered on GitHub as a signing key; commits will fail. 'ssh-add' a registered key on the host, or register one and rerun .devcontainer/setup-signing.sh."
    return 0
  fi
  if ! agent_keys | awk '{print $1" "$2}' | grep -qxF "$key"; then
    warn "the configured signing key is not loaded in the forwarded ssh-agent; run 'ssh-add' for it on the host."
    return 0
  fi
  echo "setup-signing: OK, commits will be signed with $(fingerprint "$key")"
}

if [[ "${1:-}" == "--check" ]]; then check; exit 0; fi

keys="$(agent_keys)" || die "no keys in a forwarded ssh-agent (SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-unset}). Set DC4_SSH_AGENT=1 in .devcontainer/local.env, ssh-add <key> on the host, then recreate the container: .devcontainer/dev.sh down && .devcontainer/dev.sh up"
email="$(git config --get user.email || true)"
[[ -n "$email" ]] || die "user.email is not set; set it first: git config --global user.email you@example.com"
mkdir -p "$confdir"

if [[ -n "${1:-}" ]]; then
  keys="$(grep -F -- "$1" <<<"$keys" || true)"
  [[ -n "$keys" ]] || die "no agent key matches '$1'. Available:"$'\n'"$(ssh-add -l)"
  [[ "$(wc -l <<<"$keys")" -eq 1 ]] || die "several agent keys match '$1'; be more specific:"$'\n'"$keys"
  key="$(awk '{print $1" "$2}' <<<"$keys")"
  git config --global user.signingkey "key::$key"
  git config --global --unset-all gpg.ssh.defaultKeyCommand || true
else
  refresh_registered || die "could not fetch the GitHub signing keys of '$(github_user)' (offline? set GITHUB_USER?)"
  install_helper
  git config --global --unset-all user.signingkey || true
  git config --global gpg.ssh.defaultKeyCommand "$helper"
  key="$("$helper" | sed 's/^key:://')" || {
    echo "Agent keys:" >&2; ssh-add -l >&2
    echo "Register one on GitHub as a *signing* key, then rerun this script:" >&2
    echo "  gh auth refresh -s admin:ssh_signing_key && ssh-add -L | head -n 1 | gh ssh-key add --type signing --title dc4-learning-devcontainer" >&2
    echo "  (or GitHub > Settings > SSH and GPG keys > New SSH key > Key type: Signing Key)" >&2
    exit 1
  }
fi

touch "$allowed"
grep -qF -- "$key" "$allowed" || printf '%s namespaces="git" %s\n' "$email" "$key" >> "$allowed"

git config --global gpg.format ssh
git config --global gpg.ssh.allowedSignersFile "$allowed"
git config --global commit.gpgsign true
git config --global tag.gpgsign true

# Prove the agent can actually sign with this key.
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
echo "$key" > "$tmp/key.pub"
echo test > "$tmp/msg"
ssh-keygen -Y sign -n git -f "$tmp/key.pub" -U "$tmp/msg" >/dev/null 2>&1 \
  || die "the agent refused to sign (key needs confirmation, or a hardware key needs a touch?)"

echo "setup-signing: commits and tags will now be signed with $(fingerprint "$key")"
if [[ -n "${1:-}" ]] && ! grep -qxF -- "$key" "$registered" 2>/dev/null; then
  echo "If you haven't yet, register this public key on GitHub as a *signing* key so commits show 'Verified':"
  echo "  gh auth refresh -s admin:ssh_signing_key && echo '$key' | gh ssh-key add --type signing --title dc4-learning-devcontainer"
fi

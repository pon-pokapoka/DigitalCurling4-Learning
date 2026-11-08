#!/usr/bin/env bash
# Trust a private CA in the dev container, for a TLS server that needs it.
#
# Runs on container start when there is a certificate to trust:
#   - DC4_CA_CERT: dev.sh mounts the certificate named by DC4_CA_CERT in
#     .devcontainer/local.env and sets it to the path inside the container;
#   - otherwise .devcontainer/ca.crt (not committed), which works however the
#     container is started, including VS Code "Reopen in Container".
# It never stops the container from starting.
set -uo pipefail

dest=/usr/local/share/ca-certificates/dc4-ca.crt
warn() { echo "setup-ca: WARNING: $*" >&2; exit 0; }

cert="${DC4_CA_CERT:-}"
workspace_cert="$(dirname "$0")/ca.crt"
[[ -z "$cert" && -f "$workspace_cert" ]] && cert="$workspace_cert"
[[ -n "$cert" ]] || warn "no certificate: set DC4_CA_CERT in .devcontainer/local.env or add .devcontainer/ca.crt."
[[ -r "$cert" ]] || warn "cannot read $cert."
[[ $EUID -eq 0 ]] || warn "must run as root, so the certificate was not installed."
if grep -q 'PRIVATE KEY' "$cert"; then
  warn "$cert contains a private key: skipped. Only the public certificate belongs here."
fi
[[ "$(grep -c 'BEGIN CERTIFICATE' "$cert")" -eq 1 ]] || warn "$cert must hold exactly one PEM certificate: skipped."
subject="$(openssl x509 -in "$cert" -noout -subject 2>/dev/null)" || warn "$cert is not a valid certificate: skipped."
openssl x509 -in "$cert" -noout -checkend 0 >/dev/null || echo "setup-ca: WARNING: $cert has expired." >&2

if cp "$cert" "$dest" && chmod 644 "$dest" && update-ca-certificates >/dev/null 2>&1; then
  echo "setup-ca: trusting $subject"
else
  warn "could not update the system certificates."
fi

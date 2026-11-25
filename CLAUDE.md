# DigitalCurling4-Learning — notes for Claude Code

Training and analysis for DigitalCurling4 clients. Setup is in README.md.

## Working in this repo

- **Branches.** Never push straight to `main`: branch and open a PR.
- **Public repo.** Never commit anything specific to one deployment:
  registry host, server URLs, cluster namespace and service names, SSH
  targets (e.g. `IMAGE_IMPORT_CMD`'s value), home paths, or notebook
  outputs that print them. Real values live in the uncommitted `.env`;
  committed docs and `.env.example` use placeholders like
  `registry.example`. Before each commit, grep the staged diff for
  hostnames and paths, and strip notebook outputs.
- **English only**, including docs and comments.
- **Where things live.** Match data comes from the DigitalCurling
  server's REST API, not its database. Plans, analyses and run outputs
  go under the gitignored `logs/`.
- **Self-play shares the cluster.** Don't start new matches or games
  unless asked. When told to stop, let running matches finish and start
  no new ones.

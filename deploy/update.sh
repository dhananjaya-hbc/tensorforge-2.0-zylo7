#!/usr/bin/env bash
# Pull the latest code and rebuild/restart the stack. Run on the server from the repo root:
#   deploy/update.sh
# Needs read access to the repo (a read-only deploy key, see README "Deployment").
set -euo pipefail
cd "$(dirname "$0")/.."

git pull --ff-only
echo "now at: $(git log --oneline -1)"
deploy/dc up -d --build

domain=$(grep '^DOMAIN=' .env | cut -d= -f2)
for _ in $(seq 1 60); do
  if curl -fsS "https://$domain/health" 2>/dev/null; then echo; echo "updated and healthy"; exit 0; fi
  sleep 5
done
echo "not healthy yet; check: deploy/dc logs --tail 50" >&2
exit 1

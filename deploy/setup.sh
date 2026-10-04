#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 server, run from inside the cloned repo:
#   bash deploy/setup.sh
# Re-running is safe: finished steps are skipped. Secrets are typed in and kept only in .env (mode 600).
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n==> %s\n' "$*"; }

say "swap (4 GB headroom for model loading and image builds)"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

say "Docker"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi
sudo apt-get install -y -qq unzip >/dev/null

say "settings (.env)"
if [[ ! -f .env ]]; then
  while true; do
    read -r -s -p "1/2  API key from the organizers (input hidden): " api_key; echo
    if [[ ${#api_key} -ge 16 && "$api_key" != *[[:space:]]* && "$api_key" != *.* ]]; then break; fi
    echo "     That does not look like an API key (16+ characters, no spaces or dots). Try again."
  done
  while true; do
    read -r -p "2/2  Domain for HTTPS (e.g. zylo7-sentinel.duckdns.org): " domain
    domain=$(tr '[:upper:]' '[:lower:]' <<< "$domain")
    if [[ "$domain" =~ ^[a-z0-9-]+(\.[a-z0-9-]+)+$ && "$domain" != "$api_key" ]]; then break; fi
    echo "     That is not a domain name (letters, digits, dashes and dots only). Try again."
  done
  echo "     key: ${api_key:0:4}...(${#api_key} chars)   domain: $domain"
  umask 077
  printf 'API_KEY=%s\nDOMAIN=%s\n' "$api_key" "$domain" > .env
  unset api_key
fi

domain=$(grep '^DOMAIN=' .env | cut -d= -f2)
public_ip=$(curl -fsS -m 5 https://checkip.amazonaws.com || true)
resolved=$(getent hosts "$domain" | awk '{print $1}' | head -1)
if [[ -n "$public_ip" && "$resolved" != "$public_ip" ]]; then
  echo "WARNING: $domain resolves to '${resolved:-nothing}' but this server is $public_ip; update the DNS record or HTTPS will fail." >&2
fi

say "model bundle"
if [[ ! -f models/active/model_card.json ]]; then
  read -r -p "Google Drive share link of bundle_labse_v1.0.zip: " MODEL_URL
  MODEL_URL="$MODEL_URL" sudo -E bash scripts/fetch_model.sh
  sudo chown -R "$USER:$USER" models
fi

say "build and start (first build takes a few minutes)"
sudo docker compose -f deploy/compose.prod.yml --env-file .env up -d --build

say "waiting for the model to load"
for _ in $(seq 1 60); do
  if curl -fsS "https://$domain/health" 2>/dev/null; then echo; say "live at https://$domain"; exit 0; fi
  sleep 5
done
echo "not healthy yet; check: deploy/dc ps   and   deploy/dc logs --tail 50" >&2
exit 1

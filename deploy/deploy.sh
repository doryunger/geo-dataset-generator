#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created deploy/.env from the template -- fill it in and run this again." >&2
    exit 1
fi

DOCKER=docker
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

if [ "${1:-}" != "--no-pull" ]; then
    git -C .. pull --ff-only
fi

$DOCKER compose up -d --build --force-recreate --remove-orphans

echo "Waiting for the backend to load its models and warm up..."
for _ in $(seq 1 60); do
    if curl -fs http://localhost/api/stats | grep -q '"warm": *true'; then
        token="$(curl -fs -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' || true)"
        ip="$(curl -fs -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/public-ipv4 || echo '<instance-public-ip>')"
        echo "Ready: http://$ip/"
        exit 0
    fi
    sleep 5
done

echo "Backend not ready after 5 minutes. Check: $DOCKER compose logs app" >&2
exit 1

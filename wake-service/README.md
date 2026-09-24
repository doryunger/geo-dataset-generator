# wake-service

Wakes the demo's EC2 GPU instance on first request, proxies to it once it's warm, and stops it
after `IDLE_STOP_MINUTES` of no traffic. Runs on a Lightsail box, fronted by nginx at
`refinery.stamsite.cc`. Design rationale: [context/wake-service.md](context/wake-service.md).

The real host IP, instance ID and key file aren't in this repo (it's public). The commands below
use `$LIGHTSAIL_HOST` and `$LIGHTSAIL_KEY` -- set them locally first.

## Local setup

```
cp .env.example .env   # fill in AWS creds (need ec2:DescribeInstances/StartInstances/StopInstances
                        # on the target instance) and EC2_INSTANCE_ID
docker compose up -d --build
curl http://localhost:8090/_wake/status
```

## Deploying to Lightsail

```
scp -i "$LIGHTSAIL_KEY" -r wake-service ubuntu@"$LIGHTSAIL_HOST":~/wake-service
ssh -i "$LIGHTSAIL_KEY" ubuntu@"$LIGHTSAIL_HOST"
cd ~/wake-service && cp .env.example .env   # fill in real values
docker compose up -d --build
curl http://127.0.0.1:8090/_wake/status     # confirm it's up, loopback-only
```

Then add the nginx site:

```
sudo cp nginx-site.conf /etc/nginx/sites-available/refinery.stamsite.cc
sudo ln -sf /etc/nginx/sites-available/refinery.stamsite.cc /etc/nginx/sites-enabled/refinery.stamsite.cc
sudo nginx -t && sudo systemctl reload nginx
```

and point a DNS A record for `refinery.stamsite.cc` at the Lightsail box's IP (Cloudflare-proxied,
so the origin IP stays hidden).

## Redeploying after a code change

```
scp -i "$LIGHTSAIL_KEY" main.py ubuntu@"$LIGHTSAIL_HOST":~/wake-service/main.py
ssh -i "$LIGHTSAIL_KEY" ubuntu@"$LIGHTSAIL_HOST" "cd ~/wake-service && docker compose up -d --build"
```

## Wake/stop log

Every wake, ready and stop is appended as one JSON line to `~/wake-service/logs/wake-events.jsonl`
on the Lightsail box (also printed to `docker compose logs wake`). The `stop` line summarizes the
whole session: when it woke and what triggered it (IP, IPv4/IPv6, country, user agent, path), boot
time, total time awake, and per-visitor request counts.

```
ssh -i "$LIGHTSAIL_KEY" ubuntu@"$LIGHTSAIL_HOST" "tail -n 20 ~/wake-service/logs/wake-events.jsonl"
```

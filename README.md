# IronOval — Mech Royale bot

A mechfight/1 battle royale bot for the Builders Buildathon "Builders - Mech Royale" track.

- Pure Python 3 standard library, zero dependencies, single-file bot.
- A* navigation on a grid built from the campus LiDAR terrain + building prisms.
- Zone-priority movement (always plans to stay inside the next circle).
- Lead-shot gunplay with strafing, attacker facing (rear armor), projectile/laser dodging.
- Weapon-triangle loot decisions, drone/crate farming, supply-drop capture, upgrade scoring.
- Sound tracking for blind shots and threat awareness.

## Files

| File | What |
| --- | --- |
| `Dockerfile` | Builds the bot image (`python:3.13.5-slim`, stdlib only). |
| `bot.py` | The bot (also has a `--selftest stream.jsonl.gz` replay mode). |
| `mechfight.py` | The kit's stdin/stdout protocol wrapper (unmodified). |

## Local check

```sh
python3 check/check.py -- python3 -u bot.py          # protocol conformance
docker build -t my-bot .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none \
  --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m \
  --user 65534:65534 -v "$PWD/arena:/arena:ro" my-bot
```

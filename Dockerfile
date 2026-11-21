# IronOval - mechfight/1 battle royale bot.
# Build:  docker build -t my-bot .
# Check:  python3 check/check.py --map-dir /arena -- \
#         docker run --rm -i --network none --cpus 1 --memory 512m --read-only \
#                  --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" my-bot
FROM python:3.13.5-slim
WORKDIR /bot
COPY . .
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]

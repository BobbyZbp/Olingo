#!/bin/sh
# Start everything on this Mac: Bonsai 9B (+ 4B when memory allows), then the backend, which starts the Argmax
# speech bridge. Nothing here talks to the network. Open http://localhost:8765 in Chrome.
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
# macOS refuses to open the microphone and speakers when locked ("wired") memory is close to its limit, so check first.
eval "$($PY - <<'PYEOF'
from backend import privacy
m = privacy.memory()
ours = set(privacy.llama_pids())
free = m["limit_gb"] - m["wired_gb"]
hogs = privacy.hogs(ours)
print(f'FREE_GB={free:.1f}; WIRED={m["wired_gb"]}; LIMIT={m["limit_gb"]}; HAVE_LLAMA={len(ours)}')
print('HOGS="' + ", ".join(f'{h["name"]} {h["gb"]} GB' for h in hogs) + '"')
PYEOF
)"
echo "memory: $WIRED of $LIMIT GB locked, $FREE_GB GB of headroom"
if [ "$HAVE_LLAMA" = "0" ]; then
  NEED=7.5
  if [ "$(echo "$FREE_GB < $NEED" | bc)" = "1" ]; then
    echo "WARNING: not enough headroom for both models and audio. Other apps holding memory: ${HOGS:-none found}"
    echo "         Starting in lite mode (Bonsai 9B only). Close those apps and run scripts/stop.sh && scripts/run.sh for the full setup."
    export OLINGO_LITE=1
  fi
fi
scripts/run_llms.sh
for i in $(seq 1 90); do
  curl -s -m 1 http://127.0.0.1:8082/health | grep -q ok && break
  sleep 1
done
curl -s -m 1 http://127.0.0.1:8082/health | grep -q ok || { echo "Bonsai 9B did not start; see /tmp/olingo_llm_9B.log"; exit 1; }
echo "Open http://localhost:8765"
exec $PY -m backend.app

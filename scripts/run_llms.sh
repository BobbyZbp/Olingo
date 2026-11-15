#!/bin/sh
# Bonsai-2 9B (:8082) and Bonsai-2 4B (:8084), both with their vision tower.
# The 4B handles small talk and names single objects; the 9B takes everything harder, and whatever the 4B is unsure of.
# Both listen on 127.0.0.1 only. 4096 tokens per slot: a turn is a short system prompt, one small frame (about
# 160 tokens) and the last six turns of text. The 9B has two slots: slot 0 is the conversation, whose prompt prefix
# stays cached; slot 1 runs the short helper calls (first look at the room, phrases for a new word).
# OLINGO_LITE=1 starts the 9B only (the 9B then also handles small talk, without the image).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA="$ROOT/bin/llama/llama-server"
up() { curl -s -m 1 "http://127.0.0.1:$1/health" 2>/dev/null | grep -q ok; }
if ! up 8082; then
  nohup "$LLAMA" -m "$ROOT/models/bonsai2-9B/Ternary-Bonsai-2-9B-PQ2_0.gguf" \
    --mmproj "$ROOT/models/bonsai2-9B/Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf" \
    -ngl 99 -fa on -c 8192 -np 2 --image-max-tokens 256 --cache-ram 64 \
    --host 127.0.0.1 --port 8082 > /tmp/olingo_llm_9B.log 2>&1 &
  echo "Bonsai 9B starting on :8082"
else echo "Bonsai 9B already on :8082"; fi
if [ "${OLINGO_LITE:-0}" = "1" ]; then echo "lite mode: no 4B"; exit 0; fi
if ! up 8084; then
  nohup "$LLAMA" -m "$ROOT/models/bonsai2-4B/Ternary-Bonsai-2-4B-PQ2_0.gguf" \
    --mmproj "$ROOT/models/bonsai2-4B/Ternary-Bonsai-2-4B-mmproj-Q8_0.gguf" \
    -ngl 99 -fa on -c 4096 -np 1 --image-max-tokens 256 --cache-ram 64 \
    --host 127.0.0.1 --port 8084 > /tmp/olingo_llm_4B.log 2>&1 &
  echo "Bonsai 4B starting on :8084"
else echo "Bonsai 4B already on :8084"; fi

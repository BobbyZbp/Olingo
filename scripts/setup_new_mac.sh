#!/bin/sh
# Rebuild Olingo Live on another Apple Silicon Mac (macOS 15 or later, Xcode command line tools, Chrome).
#   hf auth login                                   # the account that holds the backup
#   scripts/setup_new_mac.sh <hf-name>/olingo-backup
# Layout it creates (the code expects exactly this):
#   ~/Hackthon/olingo            this repository
#   ~/Hackthon/bin/llama         llama-server (PrismML llama.cpp build: Bonsai's PQ2_0 files need it)
#   ~/Hackthon/models            bonsai2-9B, bonsai2-4B (GGUF + vision projector)
#   ~/Hackthon/argmax-oss-swift  Argmax SDK source the speech bridge builds against
#   ~/Hackthon/.venv             Python 3.13
#   ~/Documents/huggingface/models   Argmax models (WhisperKit, TTSKit, SpeakerKit)
set -e
REPO=${1:?usage: setup_new_mac.sh <hf-name>/<dataset-name>}
H=$HOME/Hackthon
HERE=$(cd "$(dirname "$0")/.." && pwd)
[ "$(uname -m)" = "arm64" ] || { echo "Apple Silicon is required (Argmax runs on the Neural Engine)."; exit 1; }
mkdir -p "$H" "$HOME/Documents/huggingface"
[ "$HERE" = "$H/olingo" ] || { echo "Move this repository to $H/olingo first."; exit 1; }
echo "1/5 downloading the backup (about 9.5 GB)"
hf download "$REPO" --repo-type dataset --local-dir "$H/_backup"
echo "2/5 putting things in place"
[ -d "$H/models" ] || mv "$H/_backup/models" "$H/models"
[ -d "$H/bin" ] || mv "$H/_backup/bin" "$H/bin"
[ -d "$H/argmax-oss-swift" ] || mv "$H/_backup/argmax-oss-swift" "$H/argmax-oss-swift"
[ -d "$HOME/Documents/huggingface/models" ] || mv "$H/_backup/argmax-models" "$HOME/Documents/huggingface/models"
[ -d "$H/olingo/data" ] || mv "$H/_backup/data" "$H/olingo/data"
[ -d "$H/pi_backup" ] || mv "$H/_backup/pi" "$H/pi_backup"
chmod +x "$H"/bin/llama/* 2>/dev/null || true
xattr -dr com.apple.quarantine "$H/bin" 2>/dev/null || true
echo "3/5 Python environment"
command -v uv >/dev/null 2>&1 && uv venv --python 3.13 "$H/.venv" || python3 -m venv "$H/.venv"
"$H/.venv/bin/python" -m pip install -q -r "$H/olingo/requirements.txt" 2>/dev/null || uv pip install -q --python "$H/.venv/bin/python" -r "$H/olingo/requirements.txt"
echo "4/5 building the speech bridge and the Bluetooth connector"
(cd "$H/olingo/bridge" && swift build -c release)
swiftc -O "$H/olingo/bridge/bt/olingo_bt.swift" -o "$H/olingo/bridge/.build/olingo-bt"
echo "5/5 tests"
(cd "$H/olingo" && ../.venv/bin/python -m unittest discover -s tests 2>&1 | tail -1)
echo "Ready. Start with:  cd $H/olingo && scripts/run.sh   then open http://localhost:8765 in Chrome."
echo "Prove it runs offline:  scripts/offline_check.sh"
echo "The Raspberry Pi (Luma): pair it in System Settings > Bluetooth, then press Connect on the start screen."

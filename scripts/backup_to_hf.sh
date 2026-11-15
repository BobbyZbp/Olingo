#!/bin/sh
# Save everything that is not in the GitHub repository to a PRIVATE Hugging Face dataset, so the project can be
# rebuilt on another Mac after this one is cleaned up. Run once:   hf auth login   (your own account), then
#   scripts/backup_to_hf.sh <your-hf-name>/olingo-backup
# What goes up (about 10 GB): the Bonsai-2 model files, the Argmax models, the llama.cpp build, the Argmax SDK
# source the bridge was built against, the test pictures and the flashcard deck, the Raspberry Pi's programs, the
# cache of spoken sentences, the built bridge, the exact versions in use, and this repository as one file with its
# full history. Not saved: keys (github_access, pi_access), the Python environment and build folders (rebuilt).
set -e
REPO=${1:?usage: backup_to_hf.sh <hf-name>/<dataset-name>}
H=${OLINGO_HOME:-$HOME/Hackthon}
hf repo create "$REPO" --repo-type dataset --private 2>/dev/null || true
up() { [ -e "$1" ] || { echo "skipped $2 (not on this Mac)"; return 0; }; hf upload "$REPO" "$1" "$2" --repo-type dataset --quiet && echo "saved $2"; }
up "$H/models" models
up "$HOME/Documents/huggingface/models" argmax-models
up "$H/bin" bin
up "$H/argmax-oss-swift" argmax-oss-swift
up "$H/olingo/data" data
up "$H/pi_backup" pi
up "$H/olingo/cache" cache                       # every sentence already spoken (synthesised speech only, no recordings)
up "$H/olingo/bridge_bin" bridge_bin             # the speech bridge as built on this Mac
for f in RETURNING_THIS_MAC.md mac_quiet.sh mac_restore.sh; do up "$H/$f" "mac/$f"; done   # notes about this Mac
# Exact versions, so the same setup can be rebuilt: macOS, the Argmax SDK commit, every Python package.
V=$(mktemp -d)/versions.txt
{ echo "saved $(date '+%Y-%m-%d %H:%M')"; sw_vers; uname -m; echo
  echo "argmax-oss-swift: $(git -C "$H/argmax-oss-swift" remote get-url origin 2>/dev/null) at $(git -C "$H/argmax-oss-swift" rev-parse HEAD 2>/dev/null)"
  echo "olingo: $(git -C "$H/olingo" rev-parse HEAD)"; echo; echo "python $("$H/.venv/bin/python" -V 2>&1) packages:"
  "$H/.venv/bin/python" -c "import importlib.metadata as m; print('\n'.join(sorted(f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions())))" 2>/dev/null || true
} > "$V"
up "$V" versions.txt
# The repository itself with its whole history, including the notes and documents that are not on GitHub (only the
# code and the README are published there). Restore with:  git clone olingo.bundle olingo
B=$(mktemp -d)/olingo.bundle
git -C "$H/olingo" bundle create "$B" --all >/dev/null 2>&1 && up "$B" olingo.bundle
echo "Done: https://huggingface.co/datasets/$REPO (private)"

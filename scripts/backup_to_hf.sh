#!/bin/sh
# Save everything that is not in the GitHub repository to a PRIVATE Hugging Face dataset, so the project can be
# rebuilt on another Mac after this one is cleaned up. Run once:   hf auth login   (your own account), then
#   scripts/backup_to_hf.sh <your-hf-name>/olingo-backup
# What goes up (about 9.5 GB): the Bonsai-2 model files, the Argmax models, the llama.cpp build, the Argmax SDK
# source the bridge was built against, the test pictures and the flashcard deck, the Raspberry Pi's programs, and
# this repository as one file with its full history.
set -e
REPO=${1:?usage: backup_to_hf.sh <hf-name>/<dataset-name>}
H=${OLINGO_HOME:-$HOME/Hackthon}
hf repo create "$REPO" --repo-type dataset --private 2>/dev/null || true
up() { [ -e "$1" ] && hf upload "$REPO" "$1" "$2" --repo-type dataset --quiet && echo "saved $2"; }
up "$H/models" models
up "$HOME/Documents/huggingface/models" argmax-models
up "$H/bin" bin
up "$H/argmax-oss-swift" argmax-oss-swift
up "$H/olingo/data" data
up "$H/pi_backup" pi
# The repository itself with its whole history, including the notes and documents that are not on GitHub (only the
# code and the README are published there). Restore with:  git clone olingo.bundle olingo
B=$(mktemp -d)/olingo.bundle
git -C "$H/olingo" bundle create "$B" --all >/dev/null 2>&1 && up "$B" olingo.bundle
echo "Done: https://huggingface.co/datasets/$REPO (private)"

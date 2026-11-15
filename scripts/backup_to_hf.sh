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
# The helper scripts next to the keys (publishing to GitHub, reaching the Pi). The keys themselves are never saved.
for f in github_access/publish.sh github_access/use_shots.sh pi_access/ssh_config; do up "$H/$f" "mac/$f"; done
# Exact versions, so the same setup can be rebuilt: macOS, the Argmax SDK commit, every Python package.
V=$(mktemp -d)/versions.txt
{ echo "saved $(date '+%Y-%m-%d %H:%M')"; sw_vers; uname -m; echo
  echo "argmax-oss-swift: $(git -C "$H/argmax-oss-swift" remote get-url origin 2>/dev/null) at $(git -C "$H/argmax-oss-swift" rev-parse HEAD 2>/dev/null)"
  echo "olingo: $(git -C "$H/olingo" rev-parse HEAD)"; echo; echo "python $("$H/.venv/bin/python" -V 2>&1) packages:"
  "$H/.venv/bin/python" -c "import importlib.metadata as m; print('\n'.join(sorted(f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions())))" 2>/dev/null || true
} > "$V"
up "$V" versions.txt
# How to get everything back, as the first page of the dataset.
R=$(mktemp -d)/README.md
cat > "$R" <<EOF
# Olingo Live: private backup

Everything the project needs that is not in the public code repository (github.com/BobbyZbp/Olingo).
Keep this dataset private: data/ holds test pictures of people, olingo.bundle holds the internal notes.

## Continue on another Mac (Apple Silicon, macOS 15 or later, Xcode command line tools, Chrome)

    mkdir -p ~/Hackthon && cd ~/Hackthon
    hf auth login
    hf download $REPO olingo.bundle --repo-type dataset --local-dir _backup
    git clone _backup/olingo.bundle olingo
    olingo/scripts/setup_new_mac.sh $REPO
    cd olingo && scripts/run.sh          # then open http://localhost:8765 in Chrome

The bundle is the repository with its full history and its notes; start by reading the notes file in its root.
setup_new_mac.sh downloads the rest (about 10 GB), puts it in place, builds the speech bridge and runs the tests.

## What is here

- models/            Bonsai-2 9B, 4B and 2B with their vision projectors
- argmax-models/     WhisperKit, TTSKit and SpeakerKit models (go to ~/Documents/huggingface/models)
- bin/               the llama.cpp build that reads Bonsai's files
- argmax-oss-swift/  the Argmax SDK source the speech bridge is built against (commit in versions.txt)
- data/              test pictures and the flashcard deck
- cache/             sentences already spoken (synthesised speech only)
- bridge_bin/        the speech bridge as built on the first Mac
- pi/                the programs on the Raspberry Pi
- mac/               notes about the first Mac, the publishing script, the ssh settings for the Pi
- versions.txt       macOS, SDK commit and Python packages in use
- olingo.bundle      the repository, full history

## Not here, on purpose

The keys: the GitHub deploy key and the ssh key for the Pi. To publish from a new Mac, clone
git@github.com:BobbyZbp/Olingo.git as ~/Hackthon/olingo_public, add a new deploy key (or use your own GitHub
login), and adjust the paths at the top of mac/github_access/publish.sh.
EOF
up "$R" README.md
# The repository itself with its whole history, including the notes and documents that are not on GitHub (only the
# code and the README are published there). Restore with:  git clone olingo.bundle olingo
B=$(mktemp -d)/olingo.bundle
git -C "$H/olingo" bundle create "$B" --all >/dev/null 2>&1 && up "$B" olingo.bundle
echo "Done: https://huggingface.co/datasets/$REPO (private)"

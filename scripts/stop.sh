#!/bin/sh
# Stop the backend, the speech bridge and both Bonsai servers (frees about 6 GB).
pkill -f "backend.app" 2>/dev/null
pkill -f "release/OlingoBridge" 2>/dev/null
pkill -f "bin/llama/llama-server" 2>/dev/null
sleep 1
echo "stopped"

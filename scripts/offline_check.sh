#!/bin/sh
# The rule is: it must run on this Mac with WiFi off. This script proves it without anyone touching the menu bar:
# stop everything -> turn this Mac's WiFi off -> cold-start Bonsai and the Argmax bridge -> run every part of a
# call -> turn WiFi back on. WiFi is restored on any exit, and by a watchdog after 4 minutes at the latest.
cd "$(dirname "$0")/.."
DEV=$(networksetup -listallhardwareports | awk '/Wi-Fi/{getline; print $2}')
LOG=docs/offline_check.log
wifi_on() { networksetup -setairportpower "$DEV" on; }
trap wifi_on EXIT INT TERM
( sleep 240; networksetup -setairportpower "$DEV" on ) >/dev/null 2>&1 &
WATCHDOG=$!
scripts/stop.sh >/dev/null
networksetup -setairportpower "$DEV" off || { echo "could not turn WiFi off on $DEV"; exit 2; }
sleep 3
{
  echo "WiFi ($DEV): $(networksetup -getairportpower "$DEV" | awk '{print $NF}')"
  scripts/run_llms.sh
  for i in $(seq 1 90); do
    curl -s -m 1 http://127.0.0.1:8082/health | grep -q ok && curl -s -m 1 http://127.0.0.1:8084/health | grep -q ok && break
    sleep 1
  done
  ../.venv/bin/python scripts/offline_check.py
  echo "exit code $?"
} 2>&1 | tee "$LOG"
wifi_on
kill $WATCHDOG 2>/dev/null
for i in $(seq 1 40); do                       # wait until the network is really back
  curl -s -m 2 -o /dev/null https://www.apple.com && break
  sleep 1
done
echo "WiFi back on: $(networksetup -getairportpower "$DEV" | awk '{print $NF}')" | tee -a "$LOG"

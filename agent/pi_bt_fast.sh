#!/bin/sh
# Raspberry Pi: keep Bluetooth serial links out of power-saving ("sniff") mode.
# A Mac puts the link to the Pi into sniff mode, where one packet goes every 40 ms: 667 bytes x 25 a second, about
# 130 kbit/s. Without it the same link carried 830 kbit/s (measured 2026-11-15). The camera, the microphone and
# the speaker all share that link, so this is the difference between one picture a second and a smooth picture.
# Install (once, on the Pi):
#   sudo install -m 755 pi_bt_fast.sh /usr/local/bin/pi_bt_fast.sh
#   sudo sh /usr/local/bin/pi_bt_fast.sh --install        # a small service that runs this loop from boot
# Remove:  sudo systemctl disable --now pi-bt-fast && sudo rm /etc/systemd/system/pi-bt-fast.service
if [ "$1" = "--install" ]; then
  cat > /etc/systemd/system/pi-bt-fast.service <<'UNIT'
[Unit]
Description=Bluetooth serial links stay out of power-saving mode (fast link for the camera/microphone/speaker relay)
After=bluetooth.service
Wants=bluetooth.service

[Service]
ExecStart=/usr/local/bin/pi_bt_fast.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl enable --now pi-bt-fast.service
  systemctl --no-pager -n 3 status pi-bt-fast.service | head -6
  exit 0
fi
while true; do
  hciconfig hci0 lp RSWITCH 2>/dev/null                    # new connections: role switch allowed, sniff refused
  hcitool con 2>/dev/null | awk '/ACL/ {print $3, $5}' | while read addr handle; do
    if hcitool lp "$addr" 2>/dev/null | grep -q SNIFF; then  # a link made before this ran: take it out of sniff
      hcitool lp "$addr" RSWITCH >/dev/null 2>&1
      hcitool cmd 0x02 0x0004 $(printf '0x%02x 0x%02x' $((handle & 255)) $((handle >> 8))) >/dev/null 2>&1
    fi
  done
  sleep 5
done

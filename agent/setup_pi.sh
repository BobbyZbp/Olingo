#!/bin/sh
# One-time setup of the Olingo device on a Raspberry Pi 5 (Raspberry Pi OS Bookworm or later).
# Put this file and olingo_agent.py in the same folder on the Pi, then:   sudo sh setup_pi.sh
#
# What it does:
#   1. installs what is missing (camera library, ALSA tools, Bluetooth tools)
#   2. Bluetooth: serial-port service allowed, name "Olingo", visible, pairs without a code
#   3. the USB-C cable to the computer becomes a network link as well as power (after a reboot)
#   4. finds the microphone and the speaker, and starts olingo_agent.py at every boot
# Run it again any time; it only repeats what is not done yet. Not tested on real hardware yet: read the output.
set -u
[ "$(id -u)" = 0 ] || { echo "Run it with sudo:  sudo sh $0"; exit 1; }
DIR=$(cd "$(dirname "$0")" && pwd)
say() { printf '\n== %s\n' "$*"; }

say "1. packages"
NEED=""
for p in python3-picamera2 python3-pil python3-numpy alsa-utils bluez bluez-tools avahi-daemon; do
  dpkg -s "$p" >/dev/null 2>&1 || NEED="$NEED $p"
done
if [ -n "$NEED" ]; then
  echo "installing:$NEED"
  timeout 90 apt-get update -q && apt-get install -y -q $NEED || echo "!! could not install:$NEED (no internet on the Pi?). Carrying on."
else
  echo "all there"
fi

say "2. the device program"
[ -f "$DIR/olingo_agent.py" ] || { echo "!! olingo_agent.py is not next to this script"; exit 1; }
install -m 755 "$DIR/olingo_agent.py" /usr/local/bin/olingo_agent.py
echo "installed /usr/local/bin/olingo_agent.py"

say "3. Bluetooth"
BTD=/usr/libexec/bluetooth/bluetoothd
[ -x "$BTD" ] || BTD=$(command -v bluetoothd || echo /usr/sbin/bluetoothd)
mkdir -p /etc/systemd/system/bluetooth.service.d
printf '[Service]\nExecStart=\nExecStart=%s -C\n' "$BTD" > /etc/systemd/system/bluetooth.service.d/olingo.conf
cat > /etc/systemd/system/olingo-pair.service <<'UNIT'
[Unit]
Description=Olingo: accept Bluetooth pairing without a code
After=bluetooth.service
Requires=bluetooth.service

[Service]
ExecStartPre=/bin/sh -c 'rfkill unblock bluetooth; sleep 2; bluetoothctl power on; bluetoothctl system-alias Olingo; bluetoothctl discoverable-timeout 0; bluetoothctl pairable on; bluetoothctl discoverable on; true'
ExecStart=/usr/bin/bt-agent -c NoInputNoOutput
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl restart bluetooth
systemctl enable --now olingo-pair.service >/dev/null 2>&1
sleep 3
bluetoothctl show 2>/dev/null | grep -E "Name|Alias|Powered|Discoverable|Pairable" || echo "!! no Bluetooth controller found"

say "4. the USB-C cable as a network link"
BOOT=/boot/firmware
[ -f "$BOOT/config.txt" ] || BOOT=/boot
if grep -q "^dtoverlay=dwc2" "$BOOT/config.txt"; then
  echo "already on"
else
  printf '\n[all]\n# Olingo: the USB-C port is also a network link to the computer\ndtoverlay=dwc2,dr_mode=peripheral\n' >> "$BOOT/config.txt"
  grep -q "modules-load=dwc2,g_ether" "$BOOT/cmdline.txt" || sed -i 's/rootwait/rootwait modules-load=dwc2,g_ether/' "$BOOT/cmdline.txt"
  echo "switched on in $BOOT/config.txt and cmdline.txt: active after a reboot"
fi
if command -v nmcli >/dev/null 2>&1 && ! nmcli -t -f NAME con show | grep -qx olingo-usb; then
  nmcli con add type ethernet ifname usb0 con-name olingo-usb ipv4.method link-local ipv6.method link-local \
        connection.autoconnect yes >/dev/null && echo "usb0 gets a link-local address by itself"
fi
nmcli con up olingo-usb >/dev/null 2>&1 || true

say "5. microphone and speaker"
MIC=$(arecord -l 2>/dev/null | awk -F'[ :]+' '/^card/ {print "plughw:" $2 ",0"; exit}')
SPK=$(aplay -l 2>/dev/null | grep -v -i hdmi | awk -F'[ :]+' '/^card/ {print "plughw:" $2 ",0"; exit}')
[ -n "$SPK" ] || SPK=$(aplay -l 2>/dev/null | awk -F'[ :]+' '/^card/ {print "plughw:" $2 ",0"; exit}')
arecord -l 2>/dev/null | grep "^card" | sed 's/^/  microphone: /'
aplay -l 2>/dev/null | grep "^card" | sed 's/^/  speaker:    /'
[ -n "$MIC" ] || echo "!! no microphone found (plug in a USB microphone, then run this again)"
[ -n "$SPK" ] || echo "!! no speaker found"
if [ ! -f /etc/default/olingo-agent ]; then
  { echo "# Olingo device settings. Change, then: sudo systemctl restart olingo-agent"
    echo "OLINGO_ARGS=\"--bluetooth --tcp 0.0.0.0:8770 --iface usb0${MIC:+ --mic $MIC}${SPK:+ --speaker $SPK}\""; } > /etc/default/olingo-agent
fi
cat /etc/default/olingo-agent | sed 's/^/  /'

say "6. start at every boot"
cat > /etc/systemd/system/olingo-agent.service <<'UNIT'
[Unit]
Description=Olingo device: camera, microphone and speaker for Olingo
After=bluetooth.service sound.target olingo-pair.service
Wants=bluetooth.service

[Service]
EnvironmentFile=/etc/default/olingo-agent
ExecStart=/bin/sh -c 'exec /usr/bin/python3 /usr/local/bin/olingo_agent.py $OLINGO_ARGS'
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable olingo-agent.service >/dev/null 2>&1
systemctl restart olingo-agent.service
sleep 4
systemctl --no-pager -n 8 status olingo-agent.service | sed 's/^/  /'

say "done"
echo "Camera check:    rpicam-hello --list-cameras"
echo "Sound check:     arecord -D ${MIC:-plughw:1,0} -f S16_LE -r 16000 -c 1 -t raw | aplay -D ${SPK:-plughw:2,0} -f S16_LE -r 16000 -c 1 -t raw"
echo "Log:             journalctl -u olingo-agent -f"
echo "Bluetooth:       on the Mac, System Settings > Bluetooth > Olingo > Connect, then press Connect Olingo device on the page"
echo "Cable:           sudo reboot, keep the USB-C cable in the Mac, then start the Mac side with"
echo "                 OLINGO_DEVICE=tcp://$(hostname).local:8770 scripts/run.sh"

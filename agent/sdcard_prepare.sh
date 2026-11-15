#!/bin/sh
# Run on the Mac, with the Pi's SD card inserted:   agent/sdcard_prepare.sh [boot volume] [public key]
# For a Pi nobody can log in to (no screen, no keyboard, no network). It only touches the card's boot partition:
#   - config.txt, cmdline.txt: the USB-C port becomes a network link to the computer as well as power
#     (the originals are kept next to them as *.olingo-backup)
#   - ssh is switched on, and the given public key may log in (so the Mac can finish the setup over the cable)
#   - olingo/: the device program and setup_pi.sh; they are installed and run by themselves at the next boot
# Nothing on the card's system partition is read or changed from the Mac.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
B=${1:-}
KEY=${2:-$HERE/../../pi_access/olingo_pi_key.pub}
if [ -z "$B" ]; then
  for v in /Volumes/*; do
    [ -f "$v/config.txt" ] && [ -f "$v/cmdline.txt" ] && B=$v && break
  done
fi
[ -n "$B" ] && [ -f "$B/config.txt" ] && [ -f "$B/cmdline.txt" ] || { echo "No Raspberry Pi boot partition found (looked for config.txt and cmdline.txt in /Volumes/*)."; exit 1; }
[ -f "$KEY" ] || { echo "No public key at $KEY"; exit 1; }
echo "boot partition: $B"
[ -f "$B/issue.txt" ] && sed 's/^/  /' "$B/issue.txt"
if grep -q "systemd.run=" "$B/cmdline.txt" && ! grep -q "olingo_firstrun" "$B/cmdline.txt"; then
  echo "!! cmdline.txt already has a first-run script waiting (the card was written by Raspberry Pi Imager and the Pi"
  echo "   has not finished its first start). Start the Pi once, wait two minutes, then try again."
  exit 2
fi
for f in config.txt cmdline.txt; do
  [ -f "$B/$f.olingo-backup" ] || cp "$B/$f" "$B/$f.olingo-backup"
done

# 1. the USB-C port as a network link
if grep -q "^dtoverlay=dwc2,dr_mode=peripheral" "$B/config.txt"; then
  echo "config.txt: USB-C data mode was already on"
else
  printf '\n[all]\n# Olingo: the USB-C port is also a network link to the computer\ndtoverlay=dwc2,dr_mode=peripheral\n' >> "$B/config.txt"
  echo "config.txt: USB-C data mode switched on"
fi
LINE=$(tr -d '\n' < "$B/cmdline.txt")
case "$LINE" in *modules-load=dwc2,g_ether*) ;; *) LINE=$(echo "$LINE" | sed 's/rootwait/rootwait modules-load=dwc2,g_ether/') ;; esac
case "$LINE" in *modules-load=dwc2,g_ether*) ;; *) LINE="$LINE modules-load=dwc2,g_ether" ;; esac

# 2. run olingo_firstrun.sh once at the next boot (the same hook Raspberry Pi Imager uses)
RUN=/boot/firmware/olingo_firstrun.sh
grep -q "bookworm\|trixie" "$B/issue.txt" 2>/dev/null || { [ -d "$B/overlays" ] && grep -q "Raspberry Pi reference 202[0-2]" "$B/issue.txt" 2>/dev/null && RUN=/boot/olingo_firstrun.sh; }
case "$LINE" in *olingo_firstrun*) ;; *) LINE="$LINE systemd.run=$RUN systemd.run_success_action=reboot systemd.unit=kernel-command-line.target" ;; esac
printf '%s\n' "$LINE" > "$B/cmdline.txt"
echo "cmdline.txt: $(cat "$B/cmdline.txt")"

# 3. ssh on, and the files
: > "$B/ssh"
mkdir -p "$B/olingo"
cp "$HERE/olingo_agent.py" "$HERE/setup_pi.sh" "$B/olingo/"
cp "$KEY" "$B/olingo/mac_key.pub"
cat > "$B/olingo_firstrun.sh" <<'PI'
#!/bin/bash
# Olingo: runs once, as root, at the next boot (hooked in cmdline.txt); then takes its hook out and the Pi restarts.
B=/boot/firmware; [ -f $B/cmdline.txt ] || B=/boot
exec >> $B/olingo/firstrun.log 2>&1
echo "== first run, $(date)"
set -x
mkdir -p /opt/olingo
cp $B/olingo/olingo_agent.py $B/olingo/setup_pi.sh /opt/olingo/
U=$(getent passwd 1000 | cut -d: -f1)
if [ -z "$U" ]; then                 # nobody was ever set up on this Pi: a user that only this Mac's key can log in as
  U=olingo
  useradd -m -s /bin/bash "$U"
  for g in sudo video audio plugdev bluetooth netdev gpio i2c spi render input; do getent group $g >/dev/null && usermod -aG $g "$U"; done
  echo "$U ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/010_olingo
  chmod 440 /etc/sudoers.d/010_olingo
fi
H=$(getent passwd "$U" | cut -d: -f6)
install -d -m 700 -o "$U" -g "$U" "$H/.ssh"
touch "$H/.ssh/authorized_keys"
grep -qF "$(cat $B/olingo/mac_key.pub)" "$H/.ssh/authorized_keys" || cat $B/olingo/mac_key.pub >> "$H/.ssh/authorized_keys"
chown "$U:$U" "$H/.ssh/authorized_keys"
chmod 600 "$H/.ssh/authorized_keys"
echo "$U" > $B/olingo/user.txt
hostname > $B/olingo/hostname.txt
systemctl enable ssh
cat > /etc/systemd/system/olingo-setup.service <<UNIT
[Unit]
Description=Olingo: one-time device setup
After=multi-user.target bluetooth.service NetworkManager.service
ConditionPathExists=!/opt/olingo/setup-done

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'sh /opt/olingo/setup_pi.sh > $B/olingo/setup.log 2>&1; touch /opt/olingo/setup-done'
TimeoutStartSec=900

[Install]
WantedBy=multi-user.target
UNIT
systemctl enable olingo-setup.service
sed -i 's| systemd.run=[^ ]*||; s| systemd.run_success_action=[^ ]*||; s| systemd.unit=kernel-command-line.target||' $B/cmdline.txt
sync
exit 0
PI
sync
echo "files on the card:"; ls -la "$B/olingo" "$B/olingo_firstrun.sh" "$B/ssh" | sed 's/^/  /'
echo "Done. Eject the card, put it back into the Pi, and plug the Pi into the Mac with the USB-C cable."
echo "The Pi starts twice (about two minutes). Then it shows up on the Mac as a network interface."

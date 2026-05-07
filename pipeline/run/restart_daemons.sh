#!/bin/bash
# Startup: restart all live daemons after server reboot
cd /home/kemal/futures

echo "Starting HTTP server..."
# Kill existing UI server if any
pkill -f "python3 -m http.server 4173"
cd ui && setsid python3 -m http.server 4173 </dev/null &>/dev/null &
cd /home/kemal/futures

echo "Ensuring systemd services are running..."
systemctl --user restart topstepx_feed
systemctl --user restart super_structure

echo "All daemons handled."
echo "UI: http://127.0.0.1:4173"
echo "SSH tunnel: ssh -p 9909 -L 8181:127.0.0.1:4173 kemal@103.79.247.176"

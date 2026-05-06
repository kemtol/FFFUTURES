#!/bin/bash
# Startup: restart all live daemons after server reboot
cd /home/kemal/futures

echo "Starting HTTP server..."
cd ui && setsid python3 -m http.server 4173 </dev/null &>/dev/null &
cd /home/kemal/futures

echo "Starting Feed daemon..."
setsid python3 pipeline/live/run_feed.py </dev/null &>> data/Live/topstepx_feed.log &

sleep 5

echo "Starting Super Structure listener..."
setsid python3 pipeline/live/run_super_structure_live.py </dev/null &>> data/Live/super_structure.log &

echo "Starting FVG listener..."
setsid python3 -c "
from pipeline.live.fvg_scalper import FVGScalper
from datetime import datetime, timezone
s = FVGScalper()
s.check(now=datetime.now(timezone.utc))
s.run_live()
" </dev/null &>> data/Live/fvg_live.log &

sleep 2
echo "All daemons started."
echo "UI: http://127.0.0.1:4173"
echo "SSH tunnel: ssh -p 9909 -L 8181:127.0.0.1:4173 kemal@103.79.247.176"

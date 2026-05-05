#!/usr/bin/env python3
"""Super Structure live listener daemon."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
# Fallback: also try explicit path
sys.path.insert(0, "/home/kemal/futures")

from pipeline.live.tv_strategy import TVStrategy
from datetime import datetime, timezone

t = TVStrategy()
t.check(now=datetime.now(timezone.utc))
t.run_live()

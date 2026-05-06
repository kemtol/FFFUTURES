#!/usr/bin/env python3
"""Super Structure live listener daemon."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
# Fallback: also try explicit path
sys.path.insert(0, "/home/kemal/futures")

from pipeline.live.super_structure import SuperStructure
from datetime import datetime, timezone

s = SuperStructure()
s.check(now=datetime.now(timezone.utc))
s.run_live()

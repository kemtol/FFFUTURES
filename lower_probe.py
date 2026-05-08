#!/usr/bin/env python3
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOKEN_FILE = ROOT / "data" / "Live" / "topstepx_token.json"

def get_token():
    return json.loads(TOKEN_FILE.read_text())["access_token"]

def api_get(path):
    token = get_token()
    url = f"https://userapi.topstepx.com{path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

print("Testing lowercase endpoints...")
for p in ["/fill/list", "/order/list", "/position/list"]:
    try:
        res = api_get(p)
        print(f"SUCCESS: {p}")
    except:
        pass

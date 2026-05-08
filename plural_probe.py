#!/usr/bin/env python3
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOKEN_FILE = ROOT / "data" / "Live" / "topstepx_token.json"
USER_ID = 412653

def get_token():
    return json.loads(TOKEN_FILE.read_text())["access_token"]

def api_get(path):
    token = get_token()
    url = f"https://userapi.topstepx.com{path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

print("Probing plural endpoints...")
words = ["Positions", "Fills", "Orders", "Trades", "Executions", "Ledgers"]

for word in words:
    for path in [f"/{word}/all/user/{USER_ID}"]:
        try:
            res = api_get(path)
            print(f"SUCCESS: {path} -> {len(res) if isinstance(res, list) else 'Object'}")
        except Exception as e:
            # print(f"FAILED: {path} -> {e}")
            pass

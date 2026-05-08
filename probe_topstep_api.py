#!/usr/bin/env python3
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOKEN_FILE = ROOT / "data" / "Live" / "topstepx_token.json"
ACCOUNT_ID = 22303383
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

print("Probing TopstepX API for history...")
endpoints = [
    f"/Order/all/account/{ACCOUNT_ID}",
    f"/Fill/all/account/{ACCOUNT_ID}",
    f"/Position/all/account/{ACCOUNT_ID}",
    f"/Execution/all/account/{ACCOUNT_ID}",
    f"/Trade/all/account/{ACCOUNT_ID}",
]

for ep in endpoints:
    try:
        res = api_get(ep)
        print(f"SUCCESS: {ep}")
        print(f"Count: {len(res) if isinstance(res, list) else 'N/A'}")
        if res:
            print(f"Sample: {json.dumps(res[0] if isinstance(res, list) else res)[:200]}...")
    except Exception as e:
        print(f"FAILED: {ep} -> {e}")

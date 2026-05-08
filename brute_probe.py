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

print("Brute-forcing TopstepX /all/user/ endpoints...")
words = ["Position", "Fill", "Order", "Trade", "Execution", "Ledger", "Account", "Contract", "Violation"]

for word in words:
    for path in [f"/{word}/all/user/{USER_ID}", f"/{word}/list/user/{USER_ID}", f"/{word}/all", f"/{word}/list"]:
        try:
            res = api_get(path)
            print(f"SUCCESS: {path} -> {len(res) if isinstance(res, list) else 'Object'}")
            if res:
                # Save first 5 to a file for inspection
                out_file = f"api_probe_{word}.json"
                with open(out_file, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  Saved to {out_file}")
        except Exception as e:
            # print(f"FAILED: {path} -> {e}")
            pass

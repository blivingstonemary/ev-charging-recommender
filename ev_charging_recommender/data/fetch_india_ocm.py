"""Download charging stations for ALL of India from Open Charge Map (free, open data).

1. Make a free account at https://openchargemap.org and copy your API key.
2. python data/fetch_india_ocm.py YOUR_API_KEY
3. python data/prepare_india_data.py data/raw/kaggle_india.csv data/raw/ocm_india.csv

Needs internet. Uses only Python's standard library plus pandas.
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "raw" / "ocm_india.csv"


def fetch(api_key: str) -> list:
    params = {"output": "json", "countrycode": "IN", "maxresults": 20000,
              "compact": "false", "verbose": "false", "key": api_key}
    url = "https://api.openchargemap.io/v3/poi/?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "ev-charging-recommender-student-project"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def flatten(pois: list) -> pd.DataFrame:
    rows = []
    for p in pois:
        addr = p.get("AddressInfo") or {}
        conns = p.get("Connections") or []
        powers = [c.get("PowerKW") for c in conns if c.get("PowerKW")]
        types = [((c.get("ConnectionType") or {}).get("Title") or "") for c in conns]
        cost = p.get("UsageCost") or ""
        rows.append({
            "name": addr.get("Title"),
            "latitude": addr.get("Latitude"),
            "longitude": addr.get("Longitude"),
            "operator": (p.get("OperatorInfo") or {}).get("Title") or "",
            "power_kw": max(powers) if powers else None,
            "connector_types": " ; ".join(t for t in types if t),
            "price": cost if "kwh" in cost.lower() else None,
            "num_ports": p.get("NumberOfPoints") or len(conns) or 1,
            "status": (p.get("StatusType") or {}).get("Title") or "",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python data/fetch_india_ocm.py YOUR_API_KEY\n"
                         "       python data/fetch_india_ocm.py --from-json saved.json")
    pois = json.load(open(sys.argv[2], encoding="utf-8")) if sys.argv[1] == "--from-json" else fetch(sys.argv[1])
    OUT.parent.mkdir(exist_ok=True)
    df = flatten(pois)
    df.to_csv(OUT, index=False)
    print(f"Downloaded {len(df)} stations in India -> {OUT}")
    print("Next: python data/prepare_india_data.py data/raw/kaggle_india.csv data/raw/ocm_india.csv")

"""Turn one or more RAW station files into data/stations.csv (all of India).

It finds the columns by common names, keeps stations inside India, drops stations
that are not operational, cleans values, removes duplicates between files, and
fills gaps with clearly reported assumptions.

Usage (from the project folder):
    python data/prepare_india_data.py data/raw/kaggle_india.csv
    python data/prepare_india_data.py data/raw/kaggle_india.csv data/raw/ocm_india.csv
"""
import argparse
import re
from datetime import date
from pathlib import Path

import pandas as pd

# India bounding box (approx.)
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 6.0, 37.5, 68.0, 97.5

DEFAULT_POWER_KW = 22.0     # assumed when a file has no power value
DEFAULT_PRICE = 12.0        # assumed Rs per kWh when a file has no price
DEFAULT_CONNECTORS = "CCS2|Type2"

CANDIDATES = {
    "name": ["name", "station name", "station_name", "station", "location_name", "location", "title"],
    "lat": ["latitude", "lat", "y"],
    "lon": ["longitude", "lon", "lng", "long", "x"],
    "city": ["city", "city_village", "town", "district"],
    "operator": ["operator", "charge_point_operators", "network", "owner", "provider"],
    "power": ["max_power_kw", "power_kw", "power", "charger_rating", "connector_rating", "capacity", "kw"],
    "connector": ["connector_types", "connector", "connector_type", "chargers_type", "charger_type", "type", "plug", "plugs"],
    "price": ["price_per_kwh", "price", "tariff", "cost"],
    "ports": ["num_ports", "charging_points", "ports", "no_of_connectors", "connectors", "total_connectors"],
    "status": ["status", "operational_status", "state_of_station"],
}


def norm(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def find_col(df, key):
    cols = {norm(c): c for c in df.columns}
    for cand in CANDIDATES[key]:
        if norm(cand) in cols:
            return cols[norm(cand)]
    return None


def first_number(value):
    m = re.search(r"\d+(\.\d+)?", str(value))
    return float(m.group()) if m else None


def clean_connectors(value):
    text = str(value).lower()
    found = []
    if "ccs" in text:
        found.append("CCS2")
    if "chademo" in text:
        found.append("CHAdeMO")
    if "type 2" in text or "type2" in text or "iec 62196" in text or "mennekes" in text:
        found.append("Type2")
    if "gb/t" in text or "gbt" in text:
        found.append("GB/T")
    if "bharat" in text or "lev" in text or "type 1" in text or "type1" in text:
        found.append("Bharat-AC")
    return "|".join(dict.fromkeys(found)) if found else None


def read_one(path):
    raw = pd.read_csv(path, encoding_errors="replace")
    cols = {k: find_col(raw, k) for k in CANDIDATES}
    print(f"\n{path}: {len(raw)} rows")
    print("  column mapping:", {k: v for k, v in cols.items() if v})
    if not cols["lat"] or not cols["lon"]:
        raise SystemExit(f"{path}: could not find latitude/longitude columns.")

    df = pd.DataFrame()
    df["name"] = raw[cols["name"]] if cols["name"] else "Charging station"
    df["latitude"] = pd.to_numeric(raw[cols["lat"]], errors="coerce")
    df["longitude"] = pd.to_numeric(raw[cols["lon"]], errors="coerce")
    df["operator"] = raw[cols["operator"]] if cols["operator"] else ""
    df["_power"] = raw[cols["power"]].apply(first_number) if cols["power"] else None
    df["_conn"] = raw[cols["connector"]].apply(clean_connectors) if cols["connector"] else None
    df["_price"] = raw[cols["price"]].apply(first_number) if cols["price"] else None
    df["num_ports"] = raw[cols["ports"]].apply(first_number) if cols["ports"] else 1
    if cols["status"]:
        st = raw[cols["status"]].astype(str).str.lower()
        df["_bad"] = st.str.contains("not operational|planned|removed|decommission")
    else:
        df["_bad"] = False
    df["_file"] = Path(path).name
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csv", nargs="+")
    ap.add_argument("--out", default=str(Path(__file__).parent / "stations.csv"))
    args = ap.parse_args()

    df = pd.concat([read_one(p) for p in args.raw_csv], ignore_index=True)
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[~df["_bad"]]
    df = df[df["latitude"].between(LAT_MIN, LAT_MAX) & df["longitude"].between(LON_MIN, LON_MAX)].copy()
    if df.empty:
        raise SystemExit("No stations found inside India.")

    # Same place in two files (within ~10 m): keep the row that has more information.
    df["_info"] = df[["_power", "_conn", "_price"]].notna().sum(axis=1)
    df["_key"] = df["latitude"].round(4).astype(str) + "," + df["longitude"].round(4).astype(str)
    df = df.sort_values("_info", ascending=False).drop_duplicates("_key").reset_index(drop=True)

    p = df["_power"].astype(float)
    p = p.where(~(p > 1000), p / 1000)           # watts -> kW
    miss_power = p.isna() | (p <= 0)
    df["max_power_kw"] = p.where(~miss_power, DEFAULT_POWER_KW)

    miss_conn = df["_conn"].isna()
    df["connector_types"] = df["_conn"].fillna(DEFAULT_CONNECTORS)

    price = df["_price"].astype(float)
    miss_price = price.isna() | (price <= 0)
    df["price_per_kwh"] = price.where(~miss_price, DEFAULT_PRICE)

    df["num_ports"] = pd.to_numeric(df["num_ports"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    df["name"] = df["name"].fillna("Charging station").astype(str).str.strip()
    df["operator"] = df["operator"].fillna("").replace("Unknown", "")
    df.insert(0, "station_id", [f"IN{i+1:05d}" for i in range(len(df))])
    df["data_source"] = ", ".join(sorted({Path(p).name for p in args.raw_csv}))
    df["last_updated"] = date.today().isoformat()
    df["available_ports"] = pd.NA

    final = df[["station_id", "name", "operator", "latitude", "longitude", "connector_types",
                "max_power_kw", "price_per_kwh", "num_ports", "data_source",
                "last_updated", "available_ports"]]
    final.to_csv(args.out, index=False)

    n = len(final)
    print(f"\nSaved {n} stations in India to {args.out}")
    print("Assumptions used (write these in your report):")
    print(f" - power missing, set to {DEFAULT_POWER_KW} kW for {int(miss_power.sum())} of {n} stations")
    print(f" - connector missing, set to {DEFAULT_CONNECTORS} for {int(miss_conn.sum())} of {n} stations")
    print(f" - price missing, set to Rs {DEFAULT_PRICE}/kWh for {int(miss_price.sum())} of {n} stations")


if __name__ == "__main__":
    main()

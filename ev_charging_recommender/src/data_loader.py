"""Load, validate and clean the charging-station dataset."""
from __future__ import annotations
import pandas as pd

REQUIRED_COLUMNS = [
    "station_id", "name", "latitude", "longitude",
    "connector_types", "max_power_kw", "price_per_kwh",
]
OPTIONAL_COLUMNS = ["operator", "num_ports", "data_source", "last_updated", "available_ports"]


def load_stations(path: str) -> tuple[pd.DataFrame, dict]:
    """Return (clean_dataframe, report). The report documents what was removed."""
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    for c in OPTIONAL_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    report = {"rows_loaded": len(df)}

    df = df.drop_duplicates(subset="station_id")
    for c in ["latitude", "longitude", "max_power_kw", "price_per_kwh"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    report["dropped_missing_required"] = before - len(df)

    before = len(df)
    valid = (
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
        & (df["max_power_kw"] > 0)
        & (df["price_per_kwh"] >= 0)
    )
    df = df[valid]
    report["dropped_invalid_values"] = before - len(df)

    # "CCS2|Type2" -> {"ccs2", "type2"} for fast compatibility checks
    df["connector_set"] = df["connector_types"].astype(str).apply(
        lambda s: {x.strip().lower() for x in s.split("|") if x.strip()}
    )
    report["rows_usable"] = len(df)
    return df.reset_index(drop=True), report


def connector_options(df: pd.DataFrame) -> list[str]:
    """All connector names that appear in the dataset (for the UI dropdown)."""
    names: set[str] = set()
    for s in df["connector_types"].astype(str):
        names |= {x.strip() for x in s.split("|") if x.strip()}
    return sorted(names)

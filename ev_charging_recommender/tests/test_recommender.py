import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_stations
from src.recommender import UserRequest, haversine_km, recommend

# Small hand-made table used only for testing the logic (never shown in the app).
ROWS = [
    # id, name, lat, lon, connectors, kW, price, ports
    ("T1", "Near cheap slow", 9.95, 76.27, "CCS2|Type2", 7.4, 8.0, 1),
    ("T2", "Mid fast pricey", 9.99, 76.30, "CCS2", 120.0, 20.0, 6),
    ("T3", "Far fast", 10.10, 76.40, "CCS2|CHAdeMO", 60.0, 14.0, 4),
    ("T4", "CHAdeMO only", 9.97, 76.29, "CHAdeMO", 50.0, 12.0, 2),
    ("T5", "Type2 only", 9.96, 76.28, "Type2", 22.0, 10.0, 3),
]


def make_df(tmp_path):
    p = tmp_path / "s.csv"
    pd.DataFrame(ROWS, columns=["station_id", "name", "latitude", "longitude", "connector_types",
                                "max_power_kw", "price_per_kwh", "num_ports"]).to_csv(p, index=False)
    return load_stations(str(p))[0]


def test_haversine_known_distance():
    d = float(haversine_km(17.3850, 78.4867, 17.4399, 78.4983))
    assert 5 < d < 10


def test_only_compatible_connectors_returned(tmp_path):
    df = make_df(tmp_path)
    ranked, _ = recommend(df, UserRequest(9.93, 76.27, 50, "CHAdeMO"))
    assert len(ranked) > 0
    assert all("chademo" in s for s in ranked["connector_set"])


def test_priority_changes_ranking(tmp_path):
    df = make_df(tmp_path)
    base = dict(latitude=9.93, longitude=76.27, battery_pct=50, connector="CCS2")
    cheap, _ = recommend(df, UserRequest(**base, w_cost=10, w_distance=0.1, w_power=0.1, w_ports=0.1))
    fast, _ = recommend(df, UserRequest(**base, w_cost=0.1, w_distance=0.1, w_power=10, w_ports=0.1))
    assert cheap.iloc[0]["price_per_kwh"] <= fast.iloc[0]["price_per_kwh"]
    assert fast.iloc[0]["effective_kw"] >= cheap.iloc[0]["effective_kw"]


def test_low_battery_limits_range(tmp_path):
    df = make_df(tmp_path)
    ranked, notes = recommend(df, UserRequest(9.93, 76.27, 2, "CCS2"))
    assert ranked.empty or ranked["distance_km"].max() <= notes["reachable_range_km"]


def test_constant_feature_is_ignored(tmp_path):
    df = make_df(tmp_path)
    df["price_per_kwh"] = 12.0          # no price information at all
    _, notes = recommend(df, UserRequest(9.93, 76.27, 50, "CCS2", w_cost=10))
    assert notes["weights"]["cost"] == 0.0

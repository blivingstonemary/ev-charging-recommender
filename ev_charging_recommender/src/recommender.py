"""Feature engineering, KNN retrieval and preference-weighted ranking."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

EARTH_RADIUS_KM = 6371.0088


@dataclass
class UserRequest:
    latitude: float
    longitude: float
    battery_pct: float            # current state of charge, 0-100
    connector: str                # e.g. "CCS2"
    target_pct: float = 80.0      # charge up to this level
    battery_kwh: float = 40.0     # usable battery capacity
    consumption_kwh_per_km: float = 0.17
    vehicle_max_kw: float = 50.0  # max rate the car can accept
    max_distance_km: float = 25.0
    # Priority weights (any positive numbers; they are normalised to sum to 1)
    w_cost: float = 1.0
    w_distance: float = 1.0
    w_power: float = 1.0
    w_ports: float = 1.0          # more charging points = better


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km (straight line, NOT road distance)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame, req: UserRequest) -> pd.DataFrame:
    out = df.copy()
    out["distance_km"] = haversine_km(req.latitude, req.longitude,
                                      out["latitude"], out["longitude"])
    out["effective_kw"] = np.minimum(out["max_power_kw"], req.vehicle_max_kw)
    energy = max(req.target_pct - req.battery_pct, 0) / 100 * req.battery_kwh
    out["energy_kwh"] = energy
    out["est_cost"] = energy * out["price_per_kwh"]
    out["est_minutes"] = energy / out["effective_kw"] * 60
    return out


def filter_feasible(df: pd.DataFrame, req: UserRequest) -> tuple[pd.DataFrame, dict]:
    """Hard constraints: connector compatibility, radius and reachable range."""
    notes = {"start": len(df)}
    d = df[df["connector_set"].apply(lambda s: req.connector.lower() in s)]
    notes["after_connector"] = len(d)

    d = d[d["distance_km"] <= req.max_distance_km]
    notes["after_radius"] = len(d)

    # Reachable range with a 15% safety buffer (straight-line is optimistic,
    # roads are longer, so we are deliberately conservative).
    range_km = req.battery_pct / 100 * req.battery_kwh / req.consumption_kwh_per_km
    notes["reachable_range_km"] = round(range_km * 0.85, 1)
    d = d[d["distance_km"] <= range_km * 0.85]
    notes["after_range"] = len(d)
    return d.copy(), notes


def recommend(df: pd.DataFrame, req: UserRequest, top_n: int = 5,
              n_candidates: int = 20) -> tuple[pd.DataFrame, dict]:
    """Full pipeline. Returns (ranked_top_n, notes)."""
    feats = add_features(df, req)
    feasible, notes = filter_feasible(feats, req)
    if feasible.empty:
        return feasible, notes

    # --- Feature scaling (0-1). Everything is turned into a "penalty" where 0 = best.
    ports = np.log1p(pd.to_numeric(feasible["num_ports"], errors="coerce").fillna(1).clip(lower=1))
    X = np.column_stack([
        feasible["distance_km"].to_numpy(float),
        feasible["price_per_kwh"].to_numpy(float),
        feasible["effective_kw"].to_numpy(float),
        ports.to_numpy(float),
    ])
    S = MinMaxScaler().fit_transform(X)
    S[:, 2] = 1.0 - S[:, 2]            # power: 0 = fastest
    S[:, 3] = 1.0 - S[:, 3]            # ports: 0 = most charging points
    feasible[["dist_s", "cost_s", "power_s", "ports_s"]] = S

    # A factor that is identical for every station (e.g. no price data) cannot help
    # to rank, so it is ignored and the remaining weights are re-normalised.
    varies = (X.max(axis=0) - X.min(axis=0)) > 1e-9
    w = np.array([req.w_distance, req.w_cost, req.w_power, req.w_ports], float) * varies
    if w.sum() <= 0:
        w = np.array([1.0, 0, 0, 0])
    w = w / w.sum()

    # --- KNN retrieval: stations closest to the "ideal" station (all penalties 0)
    # in balanced (unweighted) feature space.
    k = min(n_candidates, len(feasible))
    use = varies.copy()
    if not use.any():                  # e.g. only one station in range
        use[0] = True
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(S[:, use])
    _, idx = knn.kneighbors(np.zeros((1, int(use.sum()))))
    cand = feasible.iloc[idx[0]].copy()

    # --- Preference-weighted multi-criteria ranking on the candidates.
    cand["pts_distance"] = w[0] * (1 - cand["dist_s"]) * 100
    cand["pts_cost"] = w[1] * (1 - cand["cost_s"]) * 100
    cand["pts_power"] = w[2] * (1 - cand["power_s"]) * 100
    cand["pts_ports"] = w[3] * (1 - cand["ports_s"]) * 100
    cand["score"] = cand[["pts_distance", "pts_cost", "pts_power", "pts_ports"]].sum(axis=1)
    cand["why"] = cand.apply(_explain, axis=1)

    ranked = cand.sort_values("score", ascending=False).head(top_n)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    notes["candidates_from_knn"] = k
    notes["weights"] = {n: float(round(v, 2)) for n, v in zip(["distance", "cost", "power", "ports"], w)}
    return ranked.reset_index(drop=True), notes


def _explain(row: pd.Series) -> str:
    parts = {"distance": row["pts_distance"], "cost": row["pts_cost"],
             "power": row["pts_power"], "charging points": row["pts_ports"]}
    best = max(parts, key=parts.get)
    return (f"Strongest factor: {best}. {row['distance_km']:.1f} km away, "
            f"{int(row['num_ports'])} charging point(s), ~{row['est_minutes']:.0f} min, "
            f"~{row['est_cost']:.0f} for this charge.")

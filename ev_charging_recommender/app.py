"""Streamlit UI (bronze + black theme).  Run:  streamlit run app.py"""
from html import escape
from pathlib import Path
import pandas as pd
import streamlit as st

from src.data_loader import load_stations, connector_options
from src.recommender import UserRequest, recommend

DATA_PATH = Path(__file__).parent / "data" / "stations.csv"
BRONZE, GOLD, DARK_BRONZE = "#CD7F32", "#E8B86D", "#8C5A2B"

st.set_page_config(page_title="EV Charging Recommender", page_icon="⚡", layout="wide")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.5rem; max-width: 1250px; }}
  [data-testid="stSidebar"] {{ border-right: 1px solid {DARK_BRONZE}; }}
  [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
      color: {GOLD}; font-size: 1rem; letter-spacing: .08em; text-transform: uppercase; }}
  .hero {{
      background: linear-gradient(120deg, #1a140d 0%, #0B0B0C 55%, #2a1d10 100%);
      border: 1px solid {DARK_BRONZE}; border-radius: 16px;
      padding: 1.6rem 2rem; margin-bottom: 1.2rem; }}
  .hero h1 {{ margin: 0; font-size: 2rem;
      background: linear-gradient(90deg, {GOLD}, {BRONZE});
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .hero p {{ margin: .3rem 0 0; color: #A89A88; font-size: .95rem; }}
  .pick {{
      background: linear-gradient(135deg, #1f1710, #120e0a);
      border: 1px solid {BRONZE}; border-radius: 16px; padding: 1.2rem 1.5rem;
      box-shadow: 0 0 24px rgba(205,127,50,.15); margin-bottom: 1rem; }}
  .pick .tag {{ color: {BRONZE}; font-size: .75rem; letter-spacing: .15em; font-weight: 700; }}
  .pick .name {{ font-size: 1.5rem; font-weight: 700; color: #F3E9DC; margin: .2rem 0 .6rem; }}
  .pick .why {{ color: #A89A88; font-size: .9rem; margin-top: .7rem; }}
  .stats {{ display: flex; gap: .8rem; flex-wrap: wrap; }}
  .stat {{ flex: 1 1 120px; background: #0B0B0C; border: 1px solid #3a2a18;
      border-radius: 12px; padding: .6rem .9rem; }}
  .stat .v {{ font-size: 1.3rem; font-weight: 700; color: {GOLD}; }}
  .stat .l {{ font-size: .72rem; color: #A89A88; text-transform: uppercase; letter-spacing: .08em; }}
  h3 {{ color: {GOLD}; }}
  hr {{ border-color: #3a2a18; }}
  div[data-testid="stExpander"] {{ border: 1px solid #3a2a18; border-radius: 12px; }}
</style>
<div class="hero">
  <h1>⚡ EV Charging Station Recommender</h1>
  <p>Find the right charger anywhere in India for your car and battery. Decision-support prototype:
  it does not guarantee availability and does not replace live navigation.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return load_stations(str(DATA_PATH))


try:
    stations, report = get_data()
except Exception as e:  # missing file / bad columns
    st.error(f"Could not load dataset: {e}")
    st.stop()

# ---------------- Sidebar: user + vehicle inputs ----------------
CITIES = {
    "Kochi": (9.9312, 76.2673), "Thiruvananthapuram": (8.5241, 76.9366),
    "Kozhikode": (11.2588, 75.7804), "Bhubaneswar": (20.2961, 85.8245),
    "Chennai": (13.0827, 80.2707), "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867), "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567), "Delhi": (28.6139, 77.2090),
    "Kolkata": (22.5726, 88.3639), "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873), "Custom location": None,
}
has_price = stations["price_per_kwh"].nunique() > 1
has_power = stations["max_power_kw"].nunique() > 1

with st.sidebar:
    st.header("📍 Location")
    city = st.selectbox("City", list(CITIES.keys()))
    if CITIES[city] is None:
        lat = st.number_input("Latitude", value=20.5937, format="%.6f")
        lon = st.number_input("Longitude", value=78.9629, format="%.6f")
    else:
        lat, lon = CITIES[city]
    max_dist = st.slider("Search radius (km)", 2, 100, 25)

    st.header("🔋 Vehicle")
    battery = st.slider("Current battery (%)", 1, 100, 40)
    target = st.slider("Charge up to (%)", 10, 100, 80)
    connector = st.selectbox("Connector type", connector_options(stations))
    battery_kwh = st.number_input("Usable battery (kWh)", 10.0, 120.0, 40.0)
    consumption = st.number_input("Consumption (kWh/km)", 0.08, 0.40, 0.17, step=0.01)
    vehicle_kw = st.number_input("Max charging rate of car (kW)", 3.0, 350.0, 50.0)

    st.header("🎯 Priorities")
    st.caption("1 = not important, 10 = very important")
    w_dist = st.slider("Shorter distance", 1, 10, 5)
    w_ports = st.slider("More charging points", 1, 10, 5)
    w_cost = st.slider("Lower cost", 1, 10, 5, disabled=not has_price,
                       help=None if has_price else "No price data in this dataset yet.")
    w_power = st.slider("Higher charging power", 1, 10, 5, disabled=not has_power,
                        help=None if has_power else "No charger-power data in this dataset yet.")
    top_n = st.slider("Stations to show", 3, 10, 5)

if target <= battery:
    st.warning("Target charge is not above the current battery level, so no energy "
               "is needed; cost and time will show as 0.")

req = UserRequest(lat, lon, battery, connector, target_pct=target,
                  battery_kwh=battery_kwh, consumption_kwh_per_km=consumption,
                  vehicle_max_kw=vehicle_kw, max_distance_km=max_dist,
                  w_cost=w_cost, w_distance=w_dist, w_power=w_power, w_ports=w_ports)
ranked, notes = recommend(stations, req, top_n=top_n)

# ---------------- Results ----------------
if ranked.empty:
    st.error("No compatible station found within your radius and reachable range. "
             "Try a larger radius, a different connector, or a higher battery level.")
    st.stop()

best = ranked.iloc[0]
st.markdown(f"""
<div class="pick">
  <div class="tag">★ BEST MATCH FOR YOU</div>
  <div class="name">{escape(str(best['name']))}</div>
  <div class="stats">
    <div class="stat"><div class="v">{best['distance_km']:.1f} km</div><div class="l">Distance</div></div>
    <div class="stat"><div class="v">{int(best['num_ports'])}</div><div class="l">Charging points</div></div>
    <div class="stat"><div class="v">{best['est_minutes']:.0f} min</div><div class="l">Est. time</div></div>
    <div class="stat"><div class="v">{best['est_cost']:.0f}</div><div class="l">Est. cost</div></div>
    <div class="stat"><div class="v">{best['score']:.0f}/100</div><div class="l">Match score</div></div>
  </div>
  <div class="why">{escape(str(best['why']))}</div>
</div>
""", unsafe_allow_html=True)

st.subheader(f"Top {len(ranked)} recommendations")
show = ranked[["rank", "name", "operator", "distance_km", "num_ports",
               "est_cost", "est_minutes", "score", "why"]].copy()
show["operator"] = show["operator"].fillna("")
show.columns = ["Rank", "Station", "Operator", "Distance (km)", "Charging points",
                "Est. cost", "Est. min", "Score", "Why recommended"]
st.dataframe(show.round(2), use_container_width=True, hide_index=True)
st.caption("Cost and time are estimates based on typical tariff and charger power.")

left, right = st.columns(2)
with left:
    st.subheader("What drove each score")
    chart = ranked.set_index("name")[["pts_distance", "pts_ports", "pts_cost", "pts_power"]]
    chart.columns = ["Distance", "Charging points", "Cost", "Power"]
    chart = chart.loc[:, (chart != 0).any()]          # hide factors with no data
    st.bar_chart(chart, color=[BRONZE, GOLD, DARK_BRONZE, "#F3E9DC"][:chart.shape[1]])
with right:
    st.subheader("Map")
    pts = ranked[["latitude", "longitude"]].copy()
    pts["c"] = BRONZE
    me = pd.DataFrame({"latitude": [lat], "longitude": [lon], "c": [GOLD]})
    st.map(pd.concat([pts, me]), color="c", size=120)
    st.caption("Gold dot = your location, bronze dots = recommended stations.")

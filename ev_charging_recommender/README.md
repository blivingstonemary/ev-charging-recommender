# AI-Based EV Charging Station Recommendation System (India)

Python · Pandas · NumPy · Scikit-learn · Streamlit

A decision-support prototype that filters EV charging stations across India for
compatibility and reach, retrieves candidates with K-Nearest Neighbors, and ranks them
by your priorities (distance, charging points, cost, charging power). It does **not**
guarantee availability and does not replace live navigation.

## Project structure
```
ev_charging_recommender/
├── app.py                        # Streamlit interface (bronze + black theme)
├── requirements.txt
├── .streamlit/config.toml        # theme colours
├── data/
│   ├── raw/kaggle_india.csv      # raw download (real stations, India)
│   ├── fetch_india_ocm.py        # optional: download all-India stations from Open Charge Map
│   ├── prepare_india_data.py     # raw file(s) -> clean data/stations.csv
│   └── stations.csv              # the dataset the app reads
├── src/
│   ├── data_loader.py            # load + validate + clean
│   └── recommender.py            # features, filtering, scaling, KNN, ranking
└── tests/test_recommender.py
```

## Steps
1. `python -m venv venv` then `venv\Scripts\activate` (Windows) and `pip install -r requirements.txt`
2. `streamlit run app.py`

## Data
`data/stations.csv` is built from `data/raw/kaggle_india.csv` (real stations, mostly Kerala
and Odisha, a few elsewhere). That file has location, operator and number of charging
points, but **no charger power, connector type or price**, so those use assumed values
(22 kW, CCS2|Type2, Rs 12/kWh). The app automatically ignores a factor that is identical for
every station, so cost and power only affect ranking once your data has real values.

To cover more of India, add Open Charge Map data (free key from openchargemap.org):
```
python data/fetch_india_ocm.py YOUR_API_KEY
python data/prepare_india_data.py data/raw/kaggle_india.csv data/raw/ocm_india.csv
```
The prepare step prints the assumptions it made; copy them into your report.

## How it works (for your report / viva)
1. **Load & validate** – required columns, numeric conversion, duplicates, invalid values.
2. **Features** – haversine distance; effective power = min(station kW, car kW); energy
   needed = (target - current)% x battery kWh; estimated cost and time.
3. **Hard filters** – connector compatibility, search radius, reachable range (15% buffer).
4. **Scaling** – MinMaxScaler to 0-1, every factor turned into a penalty where 0 is best.
5. **KNN retrieval** – NearestNeighbors finds the 20 stations closest to an "ideal station".
6. **Preference-weighted ranking** – normalised user weights give points per factor; the
   sum is the score shown in the app. Constant factors are dropped automatically.

## Limitations (state these honestly)
- Straight-line distance, not road distance or traffic.
- Power, connector and price are assumed for stations whose source has no value.
- No real-time availability; "charging points" is capacity, not free ports.
- Coverage depends on the dataset (the bundled file is concentrated in Kerala and Odisha).
- KNN here retrieves candidates; the final order comes from the weighted score.

## Tests
`python -m pip install pytest` then `python -m pytest -q tests`

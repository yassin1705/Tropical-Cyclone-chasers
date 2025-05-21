# Tropical-Cyclone-chasers

---

## 📊 Dataset Description

- **Format**: Each `.txt` file contains multiple cyclone records, each beginning with a header followed by one or more track points.
- **Fields extracted**:
  - `cyclone_id`
  - `date` and `time`
  - `LAT`, `LON` (tenths of degrees)
  - Optional: `wind_speed`, `pressure`

- **Coordinate system**:
  - Latitude and Longitude are given in **tenths of degrees** (e.g., 216 → 21.6°N)
  - Transformed to **decimal degrees** for modeling
  - Reference ellipsoid: **WGS84**

---

## 🔧 Main Features

- ✅ Parse cyclone trajectories from raw `.txt` files
- ✅ Normalize trajectories locally (per cyclone) or globally
- ✅ Convert lat/lon into **stepwise polar coordinates** using [GeographicLib](https://geographiclib.sourceforge.io/)
- ✅ Reconstruct coordinates from polar steps for error analysis
- ✅ Train models:
  - Predicting latitude & longitude **jointly**
  - Predicting latitude and longitude with **separate models**

# 🌪️ Cyclone Trajectory Modeling and Transformation

This project focuses on processing, transforming, and modeling cyclone or tornado trajectory data from historic `.txt` records. The main objective is to extract structured spatiotemporal data, apply coordinate transformations (e.g., geographic to polar), and evaluate models predicting cyclone paths.

---

## 📁 Project Structure

data/
│ └── raw/ # Raw folders of cyclone .txt files (by year)
│
notebooks/
│ └── exploratory.ipynb # Data exploration and transformation logic
│
scripts/
│ └── extract_to_csv.py # Parses .txt files into CSV format
│ └── geodesic_utils.py # GeographicLib-based step calculations
│ └── reconstruction.py # Rebuilds trajectory from polar coordinates
│
models/
│ └── model_lat_lon.py # Model predicting lat & lon jointly
│ └── model_separate.py # Separate models for lat and lon
│
README.md # Project documentation

---

## 🧪 Modeling Approaches

- **Stepwise prediction**: each next point is predicted relative to the last, using distance and bearing (geodesic).
- **Error tracking**: Reconstructed vs original points are compared with great-circle error (in meters).
- **Normalization strategies**:
  - Per-trajectory (local scale): preserves variation
  - Global scale: fixed normalization for inference

---

## 📈 Example Use

```python
from geographiclib.geodesic import Geodesic
from geodesic_utils import compute_stepwise

trajectory = [(23.4, 120.5), (23.6, 120.7), (23.9, 121.0)]
steps = compute_stepwise(trajectory)
# Output: [(2.76, 71.3), (3.58, 74.9)]  # (distance_km, bearing_deg)

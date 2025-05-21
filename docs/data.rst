Data
====

**Dataset Overview**

- **Source:** e.g. JMA best-track / IBTrACS  
- **Time span:** 1945 – 2023  
- **Spatial coverage:** Western Pacific (or “global” if applicable)  
- **Number of trajectories:** ~1 200 cyclones  
- **Record frequency:** 6-hourly best-track points  

**Fields**

| Column                 | Type    | Units / Notes                                |
|------------------------|---------|-----------------------------------------------|
| `cyclone_id`           | string  | Unique ID (e.g. “AL012023”)                   |
| `date`                 | date    | YYYY/MM/DD                                   |
| `time`                 | string  | “00Z”, “06Z”, etc.                           |
| `LAT`                  | integer | tenths of °N (e.g. 216 → 21.6°N)              |
| `LON`                  | integer | tenths of °E (0 – 360°E; wrap >180° → –180°+) |
| `wind_speed`           | int     | knots                                       |
| `pressure`             | int     | hPa                                         |
| `step_distance_km`     | float   | computed via GeographicLib                  |
| `step_bearing_deg`     | float   | relative to east (0°=E, 90°=N)              |

**Preprocessing**

1. Parse raw `.txt` → CSV  
2. Convert tenths → decimal degrees  
3. Normalize & compute (distance, bearing) steps  
4. (Optionally) reconstruct & compute error  

You can add sample code snippets or figures here as needed.

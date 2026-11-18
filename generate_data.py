"""
Generate a realistic Mumbai real estate dataset.

Why synthetic?
  Real listing sites (99acres / MagicBricks / Housing.com) block scraping
  and their CSVs go stale fast. This generator is *modelled* on public
  sources instead:
    - Kaggle: "Mumbai House Prices" / "House Prices in Mumbai"
    - Published medians from 99acres, MagicBricks & Knight Frank India
      (Mumbai price/sqft by locality, 2023-2025)
    - IGR Maharashtra registration trends (growth ~5-7%/yr, 2020 dip)

  Base Rs/sqft below are 2025 approximations, not live quotes.
  Re-run this script any time to regenerate data/mumbai_real_estate.csv.
  To use real data, replace the CSV with a Kaggle export that has the
  same column names — app.py will work unchanged.

Usage:
    python generate_data.py
    # -> data/mumbai_real_estate.csv  (1,500 rows, 2019-2025)
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

# locality, zone, base Rs/sqft in 2025, lat, lon
LOCALITIES = [
    ("Colaba", "South Mumbai", 42000, 18.9067, 72.8147),
    ("Malabar Hill", "South Mumbai", 52000, 18.9540, 72.7988),
    ("Worli", "South Mumbai", 36000, 19.0000, 72.8170),
    ("Lower Parel", "South Mumbai", 31000, 19.0000, 72.8300),
    ("Dadar West", "South Mumbai", 26500, 19.0170, 72.8470),
    ("Bandra West", "Western Suburbs", 36000, 19.0596, 72.8295),
    ("Santacruz West", "Western Suburbs", 27000, 19.0820, 72.8370),
    ("Vile Parle West", "Western Suburbs", 25000, 19.1000, 72.8450),
    ("Juhu", "Western Suburbs", 31000, 19.1070, 72.8260),
    ("Andheri West", "Western Suburbs", 21000, 19.1360, 72.8290),
    ("Andheri East", "Western Suburbs", 17500, 19.1130, 72.8690),
    ("Powai", "Eastern Suburbs", 17500, 19.1170, 72.9050),
    ("Goregaon West", "Western Suburbs", 16000, 19.1660, 72.8520),
    ("Malad West", "Western Suburbs", 15000, 19.1860, 72.8480),
    ("Borivali West", "Western Suburbs", 15800, 19.2300, 72.8550),
    ("Kandivali West", "Western Suburbs", 14200, 19.2050, 72.8500),
    ("Chembur", "Eastern Suburbs", 17800, 19.0620, 72.9000),
    ("Mulund West", "Central Suburbs", 14200, 19.1720, 72.9560),
    ("Thane West", "Thane / Navi Mumbai", 12200, 19.2180, 72.9780),
    ("Vashi", "Thane / Navi Mumbai", 13500, 19.0770, 72.9980),
]

# Year-on-year price index (2019 = 100). 2020 dip, then recovery.
YEAR_INDEX = {2019: 100, 2020: 97, 2021: 102, 2022: 110, 2023: 118, 2024: 126, 2025: 134}

AREA_BY_BHK = {
    1: (420, 650),
    2: (720, 1050),
    3: (1080, 1650),
    4: (1700, 2600),
}

FURNISHING_FACTOR = {"Unfurnished": 1.00, "Semi-Furnished": 1.04, "Fully Furnished": 1.09}
TYPE_FACTOR = {"Apartment": 1.00, "Studio": 0.94, "Villa": 1.18, "Penthouse": 1.35}


def main(n: int = 1500) -> None:
    rows = []
    for i in range(n):
        loc, zone, base_psf_2025, lat, lon = LOCALITIES[rng.integers(len(LOCALITIES))]
        year = int(rng.choice(list(YEAR_INDEX), p=[0.08, 0.09, 0.13, 0.18, 0.20, 0.17, 0.15]))
        bhk = int(rng.choice([1, 2, 3, 4], p=[0.22, 0.38, 0.28, 0.12]))
        lo, hi = AREA_BY_BHK[bhk]
        area = int(rng.integers(lo, hi + 1))
        # Penthouse / Villa only for larger homes
        if bhk >= 3 and rng.random() < 0.08:
            ptype = str(rng.choice(["Villa", "Penthouse"], p=[0.6, 0.4]))
        elif bhk == 1 and rng.random() < 0.10:
            ptype = "Studio"
        else:
            ptype = "Apartment"

        psf_year = base_psf_2025 * YEAR_INDEX[year] / YEAR_INDEX[2025]
        furnishing = str(rng.choice(list(FURNISHING_FACTOR), p=[0.45, 0.35, 0.20]))
        age = int(rng.integers(0, 26))
        floor = int(rng.integers(1, 31))
        bathrooms = int(min(4, max(1, bhk + int(rng.choice([-1, 0, 0, 1], p=[0.15, 0.55, 0.2, 0.1])))))

        # Age discount: newer = premium, older = discount (capped)
        age_factor = 1.0 - min(0.12, age * 0.004)
        noise = float(rng.lognormal(mean=0.0, sigma=0.07))  # ~ +/-7% market noise

        price = area * psf_year * FURNISHING_FACTOR[furnishing] * TYPE_FACTOR[ptype] * age_factor * noise
        price = int(round(price / 100000) * 100000)  # round to nearest lakh

        rows.append({
            "listing_id": f"MUM-{10000 + i}",
            "locality": loc,
            "zone": zone,
            "latitude": round(lat + float(rng.normal(0, 0.004)), 4),
            "longitude": round(lon + float(rng.normal(0, 0.004)), 4),
            "year": year,
            "month": int(rng.integers(1, 13)),
            "bhk": bhk,
            "bathrooms": bathrooms,
            "area_sqft": area,
            "floor": floor,
            "age_years": age,
            "property_type": ptype,
            "furnishing": furnishing,
            "price_inr": price,
            "price_per_sqft": int(round(price / area)),
        })

    df = pd.DataFrame(rows).sort_values(["year", "locality"]).reset_index(drop=True)
    out = Path(__file__).parent / "data" / "mumbai_real_estate.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows -> {out}")
    print(df.groupby("year")["price_per_sqft"].median().round(0).to_string())
    print(f"Median price: Rs {df['price_inr'].median():,.0f}  |  "
          f"Median Rs/sqft: {df['price_per_sqft'].median():,.0f}")


if __name__ == "__main__":
    main()

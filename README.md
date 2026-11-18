# 🏙️ Mumbai Real Estate Trends

An end-to-end Python + Streamlit web app that analyses Mumbai property
prices — with interactive filters, clean charts, a map, and a
machine-learning price predictor.

**Live demo:** deploy it in ~10 minutes with the steps below (free on
Streamlit Community Cloud).

![Stack](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-red)
![Model](https://img.shields.io/badge/Model-RandomForest-green)

---

## What it does

| Tab | What you get |
| --- | --- |
| **📊 Overview** | KPIs (median price, ₹/sqft, listings), price trend 2019–2025, ₹/sqft by locality, price-vs-area scatter, price distribution |
| **📍 Trends & Map** | ₹/sqft by zone over time, median price by BHK, interactive map of listings, cheapest vs priciest localities |
| **🤖 Price Predictor** | Random Forest model — pick locality, BHK, area, age, floor, type & furnishing and get an estimated price + typical range. Shows R², MAE and feature importance |
| **📋 Data** | Filtered listings table + **Download filtered CSV** button |

**Filters (sidebar, apply everywhere):** zone, locality, BHK, property type,
year, price (₹ Cr).

---

## Dataset — realistic, modelled, honest

`data/mumbai_real_estate.csv` — **1,500 listings, 20 localities, 2019–2025**.

Real listing sites block scraping and their data goes stale fast, so this
dataset is *modelled* on public sources rather than scraped:

- **Kaggle** — “Mumbai House Prices” / “House Prices in Mumbai”
- **Published medians** — 99acres, MagicBricks and Knight Frank India
  (Mumbai ₹/sqft by locality, 2023–2025)
- **IGR Maharashtra registration trends** — ~5–7% annual growth, 2020 dip

Base 2025 ₹/sqft used in the generator (see `generate_data.py`):
Malabar Hill ~₹52,000 · Colaba ~₹42,000 · Worli / Bandra West ~₹36,000 ·
Juhu ~₹31,000 · Dadar / Santacruz ~₹26–27,000 · Andheri West ~₹21,000 ·
Powai / Chembur ~₹17–18,000 · Thane West ~₹12,200.

> ⚠️ This is a **learning / demo dataset, not live listings** and not
> investment advice. Prices are approximations.

### Regenerate the data

```bash
python generate_data.py
# -> data/mumbai_real_estate.csv
```

### Swap in real data

Download a Kaggle Mumbai house-prices CSV, rename its columns to match
`data/mumbai_real_estate.csv` (`locality, zone, latitude, longitude, year,
month, bhk, bathrooms, area_sqft, floor, age_years, property_type,
furnishing, price_inr, price_per_sqft`) and replace the file — `app.py`
works unchanged.

---

## Project structure

```
mumbai-real-estate/
├── app.py                  # The whole Streamlit app (load → filter → chart → predict)
├── generate_data.py        # Builds data/mumbai_real_estate.csv (seeded, reproducible)
├── data/
│   └── mumbai_real_estate.csv
├── requirements.txt
├── .streamlit/config.toml  # Theme
└── README.md
```

### How the code works (step by step)

1. **`load_data()`** — `@st.cache_data` reads the CSV once; adds `price_cr`.
2. **Sidebar filters** — multiselects / sliders build one boolean mask;
   `filtered` is the dataframe every chart uses.
3. **KPIs + charts** — Plotly Express: line (trend), bar (locality / BHK),
   scatter (price vs area), histogram (distribution), `st.map` (locations).
4. **`train_model()`** — `@st.cache_resource` trains a
   `RandomForestRegressor(n_estimators=200)` on area, BHK, bathrooms, age,
   floor, year + one-hot locality / type / furnishing. 80/20 split,
   reports **R²** and **MAE**.
5. **`predict_price()`** — one-hot-encodes the form inputs, reindexes them
   to the training columns, and calls `model.predict()`.

---

## Run locally

```bash
# 1. Python 3.10+ required
python --version

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
# -> open http://localhost:8501 in your browser
```

### Verified locally (2026-11-18)

- `python -m py_compile app.py generate_data.py` — OK
- Data: 1,500 rows, 20 localities, 2019–2025, 0 nulls; median ₹1.78 Cr
- Model (Random Forest, 200 trees, 80/20 split): **R² = 0.922, MAE = ₹0.29 Cr**
- Sample check: Andheri West, 2 BHK, 950 sqft, 5 yrs old → **₹1.71 Cr**
- `streamlit run app.py` (Streamlit 1.67.0) — serves **HTTP 200** on localhost

---

## Deploy to a live URL (free, ~10 min)

### Option A — Streamlit Community Cloud (recommended)

1. Create a GitHub repo, e.g. `mumbai-real-estate`, and push this folder:
   ```bash
   git init
   git add .
   git commit -m "Mumbai real estate Streamlit app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/mumbai-real-estate.git
   git push -u origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → sign in with GitHub → **Create app**.
3. Pick repo `mumbai-real-estate`, branch `main`, main file `app.py` → **Deploy**.
4. Wait 2–3 minutes. You get a live URL like:
   `https://mumbai-real-estate-<your-username>.streamlit.app`
5. Share that link — anyone can open it, no login needed.

> If deploy fails, check the Cloud logs: 99% of the time it is a missing
> package in `requirements.txt` or a wrong main-file path.

### Option B — Hugging Face Spaces (alternative, also free)

1. huggingface.co → New Space → SDK: **Streamlit** → upload these files.
2. It builds automatically and gives you a `*.hf.space` URL.

---

## Try these first

1. Filter to **Bandra West, 2 BHK, 2023–2025** — what is the median ₹/sqft?
2. Compare **South Mumbai vs Western Suburbs** on the Trends tab.
3. In the Predictor, keep everything fixed and change only the **locality**
   from Thane West → Malabar Hill — how much does the estimate jump?
4. Download the filtered CSV and open it in Excel / Google Sheets.

## Next ideas

- Add a real Kaggle dataset (see “Swap in real data” above)
- Add a train/delay version: same app shape, Mumbai suburban trains
  (Central / Western / Harbour lines, station, delay minutes)
- Add login / saved predictions, or a monthly price-index page

## Licence

MIT — free to use, modify and share for learning.

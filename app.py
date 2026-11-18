"""
Mumbai Real Estate Trends — Streamlit app
==========================================
End-to-end data app: load -> filter -> visualise -> predict.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy (free): push this folder to GitHub, then
    streamlit.io/cloud -> New app -> pick repo / app.py -> Deploy.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/mumbai_real_estate.csv"

st.set_page_config(
    page_title="Mumbai Real Estate Trends",
    page_icon="🏙️",
    layout="wide",
)


# ---------------------------------------------------------------- helpers
def inr(value: float) -> str:
    """Format rupees as ₹1.78 Cr / ₹85.5 Lakh — how Mumbai prices are quoted."""
    if value >= 10_000_000:
        return f"₹{value / 10_000_000:.2f} Cr"
    return f"₹{value / 100_000:.1f} Lakh"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["price_cr"] = df["price_inr"] / 10_000_000
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train once, reuse on every interaction. Returns model + metrics."""
    X = pd.get_dummies(
        df[["area_sqft", "bhk", "bathrooms", "age_years", "floor", "year",
            "locality", "property_type", "furnishing"]],
        columns=["locality", "property_type", "furnishing"],
        drop_first=True,
    )
    y = df["price_inr"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "columns": list(X.columns),
        "importance": pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False).head(10),
    }
    return model, metrics


def predict_price(model, columns, inputs: dict) -> float:
    row = pd.DataFrame([inputs])
    row = pd.get_dummies(row, columns=["locality", "property_type", "furnishing"])
    row = row.reindex(columns=columns, fill_value=0)
    return float(model.predict(row)[0])


# ---------------------------------------------------------------- data + filters
df = load_data()

st.sidebar.header("🔍 Filters")
st.sidebar.caption("Filters apply to every chart, KPI and table below — "
                   "except the predictor, which trains on the full dataset.")

zones = st.sidebar.multiselect(
    "Zone", sorted(df["zone"].unique()), default=sorted(df["zone"].unique()))
localities = st.sidebar.multiselect(
    "Locality", sorted(df[df["zone"].isin(zones)]["locality"].unique()),
    default=sorted(df[df["zone"].isin(zones)]["locality"].unique()))
bhk_sel = st.sidebar.multiselect(
    "BHK", sorted(df["bhk"].unique()), default=sorted(df["bhk"].unique()))
ptype_sel = st.sidebar.multiselect(
    "Property type", sorted(df["property_type"].unique()),
    default=sorted(df["property_type"].unique()))
year_min, year_max = st.sidebar.slider(
    "Year", int(df["year"].min()), int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max())))
price_min, price_max = st.sidebar.slider(
    "Price (₹ Cr)", 0.2, 12.0, (0.2, 12.0), step=0.1)

filtered = df[
    df["zone"].isin(zones)
    & df["locality"].isin(localities)
    & df["bhk"].isin(bhk_sel)
    & df["property_type"].isin(ptype_sel)
    & df["year"].between(year_min, year_max)
    & df["price_cr"].between(price_min, price_max)
]

if filtered.empty:
    st.warning("No listings match those filters — widen the price, year or locality range.")
    st.stop()

# ---------------------------------------------------------------- header
st.title("🏙️ Mumbai Real Estate Trends")
st.markdown(
    "Interactive analysis of **1,500 listings across 20 Mumbai localities, 2019–2025** — "
    "prices, price/sqft, and a machine-learning price predictor. "
    "Use the sidebar to filter by zone, locality, BHK, type, year and price."
)
st.caption(
    "⚠️ Data note: realistic dataset *modelled* on public sources "
    "(Kaggle Mumbai house prices; published 99acres / MagicBricks / Knight Frank medians). "
    "It is for learning and demo — not live listings or investment advice. "
    "See README.md for sources and how to swap in real data."
)

# ---------------------------------------------------------------- KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings shown", f"{len(filtered):,}", f"of {len(df):,} total")
c2.metric("Median price", inr(filtered["price_inr"].median()))
c3.metric("Median ₹/sqft", f"₹{filtered['price_per_sqft'].median():,.0f}")
c4.metric("Median area", f"{filtered['area_sqft'].median():,.0f} sqft")

# Year-on-year growth on the filtered set (needs 2+ years)
trend = filtered.groupby("year").agg(
    median_psf=("price_per_sqft", "median"),
    median_price=("price_inr", "median"),
    listings=("listing_id", "count"),
).reset_index()
if len(trend) >= 2:
    growth = (trend["median_psf"].iloc[-1] / trend["median_psf"].iloc[0] - 1) * 100
    st.caption(f"📈 Median ₹/sqft changed **{growth:+.1f}%** over the selected years "
               f"({trend['year'].iloc[0]}–{trend['year'].iloc[-1]}).")

# ---------------------------------------------------------------- tabs
tab_overview, tab_trends, tab_predict, tab_data = st.tabs(
    ["📊 Overview", "📍 Trends & Map", "🤖 Price Predictor", "📋 Data"])

with tab_overview:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Median ₹/sqft over time")
        fig = px.line(trend, x="year", y="median_psf", markers=True,
                      labels={"median_psf": "Median ₹/sqft", "year": "Year"})
        fig.update_layout(yaxis_tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Median ₹/sqft by locality")
        by_loc = (filtered.groupby("locality")["price_per_sqft"].median()
                  .sort_values().reset_index())
        fig = px.bar(by_loc, x="price_per_sqft", y="locality", orientation="h",
                     color="price_per_sqft", color_continuous_scale="Blues",
                     labels={"price_per_sqft": "Median ₹/sqft"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price vs area — bigger homes, higher total price?")
    fig = px.scatter(filtered, x="area_sqft", y="price_inr", color="bhk",
                     hover_data=["locality", "property_type", "year"],
                     labels={"area_sqft": "Area (sqft)", "price_inr": "Price (₹)",
                             "bhk": "BHK"},
                     color_continuous_scale="Viridis")
    fig.update_layout(yaxis_tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price distribution")
    fig = px.histogram(filtered, x="price_cr", nbins=30,
                       labels={"price_cr": "Price (₹ Cr)"},
                       title="Most listings cluster between ₹1–3 Cr")
    st.plotly_chart(fig, use_container_width=True)

with tab_trends:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("₹/sqft by zone over time")
        by_zone = (filtered.groupby(["year", "zone"])["price_per_sqft"].median()
                   .reset_index())
        fig = px.line(by_zone, x="year", y="price_per_sqft", color="zone",
                      markers=True, labels={"price_per_sqft": "Median ₹/sqft"})
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.subheader("Median price by BHK")
        by_bhk = (filtered.groupby("bhk")["price_inr"].median().reset_index())
        by_bhk["price_cr"] = by_bhk["price_inr"] / 10_000_000
        fig = px.bar(by_bhk, x="bhk", y="price_cr", text_auto=".2f",
                     labels={"bhk": "BHK", "price_cr": "Median price (₹ Cr)"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Where are the listings? (map)")
    st.map(filtered[["latitude", "longitude"]].dropna(), zoom=10)
    st.caption("Each dot is a listing in the filtered set. Zoom / drag to explore "
               "South Mumbai vs the Western and Central suburbs.")

    st.subheader("Cheapest vs priciest localities")
    summary = (filtered.groupby(["locality", "zone"])
               .agg(listings=("listing_id", "count"),
                    median_price=("price_inr", "median"),
                    median_psf=("price_per_sqft", "median"))
               .reset_index().sort_values("median_psf"))
    summary["median_price"] = summary["median_price"].apply(inr)
    summary["median_psf"] = summary["median_psf"].apply(lambda v: f"₹{v:,.0f}")
    st.dataframe(summary, use_container_width=True, hide_index=True)

with tab_predict:
    st.subheader("🤖 Predict a Mumbai home price")
    st.markdown("A **Random Forest** model trained on all 1,500 listings. "
                "Pick a home below — the model uses area, BHK, locality, age, "
                "floor, type and furnishing to estimate its price.")
    model, metrics = train_model(df)

    m1, m2 = st.columns(2)
    m1.metric("Model accuracy (R²)", f"{metrics['r2']:.2f}",
              "1.0 = perfect, 0 = no better than the average")
    m2.metric("Typical error (MAE)", inr(metrics["mae"]),
              "average |predicted − actual| on unseen listings")

    p1, p2, p3 = st.columns(3)
    with p1:
        in_locality = st.selectbox("Locality", sorted(df["locality"].unique()),
                                   index=sorted(df["locality"].unique()).index("Andheri West"))
        in_bhk = st.selectbox("BHK", [1, 2, 3, 4], index=1)
        in_area = st.slider("Area (sqft)", 400, 2600, 950, step=10)
    with p2:
        in_bath = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
        in_age = st.slider("Age of building (years)", 0, 25, 5)
        in_floor = st.slider("Floor", 1, 30, 8)
    with p3:
        in_type = st.selectbox("Property type", sorted(df["property_type"].unique()))
        in_furn = st.selectbox("Furnishing", sorted(df["furnishing"].unique()))
        in_year = st.selectbox("Year (market year)", sorted(df["year"].unique()),
                               index=len(sorted(df["year"].unique())) - 1)

    if st.button("Predict price", type="primary"):
        inputs = {"area_sqft": in_area, "bhk": in_bhk, "bathrooms": in_bath,
                  "age_years": in_age, "floor": in_floor, "year": in_year,
                  "locality": in_locality, "property_type": in_type,
                  "furnishing": in_furn}
        price = predict_price(model, metrics["columns"], inputs)
        st.success(f"**Estimated price: {inr(price)}** "
                   f"(≈ ₹{price / in_area:,.0f}/sqft)")
        st.info(f"Typical range, given the model's average error: "
                f"**{inr(max(0, price - metrics['mae']))} – "
                f"{inr(price + metrics['mae'])}**")
        st.caption("This is a demo estimate from a modelled dataset, not a valuation. "
                   "Real pricing also depends on building, view, amenities and negotiation.")

    st.subheader("What drives price most?")
    imp = metrics["importance"].reset_index()
    imp.columns = ["feature", "importance"]
    fig = px.bar(imp, x="importance", y="feature", orientation="h",
                 labels={"importance": "Feature importance"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Area and locality dominate — as expected in Mumbai. "
               "Try changing only the locality in the form above and re-predicting.")

with tab_data:
    st.subheader("Filtered listings")
    show = filtered.sort_values("price_inr", ascending=False)
    st.dataframe(show[["listing_id", "locality", "zone", "year", "bhk", "bathrooms",
                       "area_sqft", "property_type", "furnishing",
                       "price_inr", "price_per_sqft"]],
                 use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download filtered data (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="mumbai_real_estate_filtered.csv",
        mime="text/csv",
    )
    st.markdown("""
    **Columns**
    - `locality`, `zone` — where the home is · `latitude`, `longitude` — for the map
    - `year`, `month` — listing year · `bhk`, `bathrooms`, `area_sqft`, `floor`, `age_years`
    - `property_type` — Apartment / Studio / Villa / Penthouse · `furnishing`
    - `price_inr`, `price_per_sqft` — total price and price per sqft
    """)

st.divider()
st.caption("Built with Python + Streamlit + scikit-learn + Plotly · "
           "Dataset: data/mumbai_real_estate.csv (regenerate with `python generate_data.py`)")

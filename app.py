# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests
from io import BytesIO
from pathlib import Path

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

# =========================================================
# Page config & tiny CSS
# =========================================================
st.set_page_config(page_title="Restaurant Rating Predictor", page_icon="🍽️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.stMetric {background: #ffffff20; border-radius: 0.5rem; padding: 0.75rem;}
[data-testid="stSidebar"] {background-color: #f5f5f7;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 0. Data source (Google Drive direct link)
# =========================================================
DATA_URL = "https://drive.google.com/uc?export=download&id=1iRWeGaDybybQ2eiTCyEgyXDYbH5FpFup"

# =========================================================
# 1. Load data
# =========================================================
@st.cache_data(show_spinner="Downloading data from Google Drive...")
def load_data(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        df = pd.read_csv(BytesIO(r.content))
    except Exception as e:
        st.warning(f"Failed to download/read your data: {e}\nShowing a small demo dataset instead.")
        df = pd.DataFrame({
            "name": ["Demo A", "Demo B", "Demo C"],
            "city": ["SF", "SF", "SF"],
            "Geographic Area Name": ["ZIP 94102", "ZIP 94103", "ZIP 94104"],
            "category": ["Asian", "Cafe", "Italian"],
            "price": [2, 1, 3],
            "popularity": [0.8, 0.5, 0.9],
            "sentiment": [0.2, -0.1, 0.5],
            "num_reviews": [120, 45, 300],
            "google_rating": [4.6, 4.2, 4.8],
            "mean_income": [85000, 90000, 110000],
            "unemployment_percent": [4.5, 3.8, 2.9],
            "poverty_percent": [10.2, 9.1, 6.7],
            "latitude": [37.7749, 37.7849, 37.7649],
            "longitude": [-122.4194, -122.4094, -122.4294],
        })
    return df

df_raw = load_data(DATA_URL).copy()

# =========================================================
# 2. Preprocess + Train models (cached)
# =========================================================
@st.cache_resource(show_spinner="Training models (only runs once)...")
def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Postal code extraction (optional if already present)
    if "postal_code" not in df.columns and "Geographic Area Name" in df.columns:
        df["postal_code"] = df["Geographic Area Name"].astype(str).str.extract(r"(\d{5})")

    # Targets
    if "google_rating" not in df.columns:
        raise ValueError("Column 'google_rating' is required for training the regression model.")
    df["highly_rated"] = (df["google_rating"] > 4.5).astype(int)

    required = [
        "price", "popularity", "sentiment", "num_reviews", "category", "google_rating",
        "mean_income", "unemployment_percent", "poverty_percent", "latitude", "longitude"
    ]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # Bin reviews
    q1, q2 = df["num_reviews"].quantile([0.33, 0.66])
    df["reviews_bin"] = pd.cut(
        df["num_reviews"],
        bins=[-1, q1, q2, df["num_reviews"].max()],
        labels=["low", "medium", "high"]
    )

    # ---------- Feature sets ----------
    # Regression
    num_feats_reg = ["price", "popularity", "sentiment"]
    cat_feats_reg = ["reviews_bin", "category"]
    X_reg = df[num_feats_reg + cat_feats_reg]
    y_reg = df["google_rating"]

    # Classification
    num_feats_clf = ["mean_income", "unemployment_percent", "poverty_percent",
                     "price", "popularity", "sentiment"]
    cat_feats_clf = ["reviews_bin", "category"]
    X_clf = df[num_feats_clf + cat_feats_clf]
    y_clf = df["highly_rated"]

    prep_reg = ColumnTransformer([
        ("num", StandardScaler(), num_feats_reg),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats_reg)
    ])
    prep_clf = ColumnTransformer([
        ("num", StandardScaler(), num_feats_clf),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats_clf)
    ])

    # ---------- 3) Regression: pick best via CV ----------
    models_reg = {
        "DecisionTree": (
            DecisionTreeRegressor(random_state=42),
            {"reg__max_depth": [None, 5, 10], "reg__min_samples_split": [2, 5, 10]}
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {"reg__n_estimators": [200], "reg__max_depth": [None, 10, 20]}
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {"reg__n_estimators": [200], "reg__learning_rate": [0.05, 0.1]}
        )
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_rmse = float("inf")
    best_model_name = None
    best_params_reg = None

    for name, (est, params) in models_reg.items():
        pipe = Pipeline([("preproc", prep_reg), ("reg", est)])
        gs = GridSearchCV(pipe, params, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
        gs.fit(X_reg, y_reg)
        mse = mean_squared_error(y_reg, gs.predict(X_reg))
        rmse = mse ** 0.5
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_params_reg = gs.best_params_

    # Fit final regressor
    best_regressor = models_reg[best_model_name][0]
    best_reg_pipe = Pipeline([("preproc", prep_reg), ("reg", best_regressor)])
    best_reg_pipe.set_params(**best_params_reg)
    best_reg_pipe.fit(X_reg, y_reg)

    # ---------- 4) Classification stacking ----------
    rf_pipe = Pipeline([
        ("preproc", prep_clf),
        ("clf", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ])
    gb_pipe = Pipeline([
        ("preproc", prep_clf),
        ("clf", GradientBoostingClassifier(random_state=42))
    ])

    cal_rf = CalibratedClassifierCV(rf_pipe, cv=5, method="sigmoid"); cal_rf.fit(X_clf, y_clf)
    cal_gb = CalibratedClassifierCV(gb_pipe, cv=5, method="sigmoid"); cal_gb.fit(X_clf, y_clf)

    stack = StackingClassifier(
        estimators=[("rf", cal_rf), ("gb", cal_gb)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, stack_method="predict_proba"
    )

    param_grid_clf = {
        "final_estimator__C": [0.01, 0.1, 1, 10],
        "final_estimator__class_weight": [None, "balanced"]
    }
    gs_clf = GridSearchCV(stack, param_grid_clf, scoring="precision", cv=5, n_jobs=-1)
    gs_clf.fit(X_clf, y_clf)
    best_clf = gs_clf.best_estimator_
    best_clf.fit(X_clf, y_clf)

    # ---------- 5) Predictions ----------
    df["predicted_rating"] = best_reg_pipe.predict(X_reg).round(1)
    df["probability"] = best_clf.predict_proba(X_clf)[:, 1]

    # Return full df with preds
    return df, best_rmse, best_model_name

df, reg_rmse, reg_name = train_and_predict(df_raw)

# Choose rating column to display
actual_rating_col = "google_rating" if "google_rating" in df.columns else None

# =========================================================
# 3. Sidebar filters
# =========================================================
st.sidebar.title("Filters")

# Toggle which rating to color by
use_pred = st.sidebar.radio("Color points by", ["Predicted rating", "Actual rating" if actual_rating_col else "Predicted rating only"]) == "Predicted rating"

# Category
if "category" in df.columns:
    cat_options = sorted(df["category"].dropna().astype(str).unique())
    selected_categories = st.sidebar.multiselect("Category", cat_options)
else:
    selected_categories = []

# Price
if "price" in df.columns and pd.api.types.is_numeric_dtype(df["price"]):
    pmin, pmax = int(df["price"].min()), int(df["price"].max())
    price_range = st.sidebar.slider("Price range", pmin, pmax, (pmin, pmax))
else:
    price_range = None

# Probability threshold
prob_threshold = st.sidebar.slider("Min probability (highly rated)", 0.0, 1.0, 0.0, 0.01)

# Build mask
mask = pd.Series(True, index=df.index)
if selected_categories and "category" in df.columns:
    mask &= df["category"].astype(str).isin(selected_categories)
if price_range and "price" in df.columns:
    mask &= df["price"].between(*price_range)
mask &= df["probability"] >= prob_threshold

show_df = df.loc[mask].copy()

# =========================================================
# 4. KPIs
# =========================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Records", len(show_df))
if actual_rating_col:
    col2.metric("Avg. Actual Rating", f"{show_df[actual_rating_col].mean():.2f}")
else:
    col2.metric("Avg. Actual Rating", "N/A")
col3.metric("Avg. Predicted Rating", f"{show_df['predicted_rating'].mean():.2f}")
col4.metric("Avg. Probability", f"{show_df['probability'].mean():.2f}")

st.caption(f"Best regressor: **{reg_name}**, RMSE on full data (approx): **{reg_rmse:.3f}**")

# =========================================================
# 5. Map
# =========================================================
st.subheader("Interactive Map")

if {"latitude", "longitude"}.issubset(show_df.columns):
    # color by rating metric
    color_source = show_df["predicted_rating"] if use_pred or not actual_rating_col else show_df[actual_rating_col]
    rmin, rmax = color_source.min(), color_source.max()

    def rating_to_rgb(r):
        if pd.isna(r) or rmax == rmin:
            return [180, 180, 180, 170]
        t = (r - rmin) / (rmax - rmin)  # 0~1
        # blue -> green -> red gradient
        return [
            int(255 * (1 - t)),      # R
            int(100 + 155 * t),      # G
            int(200 * (1 - t)),      # B
            180
        ]

    show_df["color"] = color_source.apply(rating_to_rgb)

    initial_view = pdk.ViewState(
        latitude=float(show_df["latitude"].mean()),
        longitude=float(show_df["longitude"].mean()),
        zoom=11
    )

    tooltip_text = "{name}\\nPred: {predicted_rating}\\nProb: {probability}"
    if actual_rating_col:
        tooltip_text += f"\\nActual: {{{actual_rating_col}}}"
    if "category" in show_df.columns:
        tooltip_text += "\\nCat: {category}"

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=show_df,
        get_position='[longitude, latitude]',
        get_radius=60,
        get_fill_color="color",
        pickable=True,
    )

    show_density = st.toggle("Show density (HexagonLayer)", value=False)
    layers = [scatter]
    if show_density:
        hex_layer = pdk.Layer(
            "HexagonLayer",
            data=show_df,
            get_position='[longitude, latitude]',
            radius=130,
            elevation_scale=4,
            elevation_range=[0, 1000],
            extruded=True,
            coverage=0.9,
            pickable=True
        )
        layers.insert(0, hex_layer)

    deck = pdk.Deck(initial_view_state=initial_view, layers=layers, tooltip={"text": tooltip_text})
    st.pydeck_chart(deck)
else:
    st.error("Missing 'latitude' and 'longitude' columns.")

# =========================================================
# 6. Table & Download
# =========================================================
with st.expander("Show data table"):
    st.dataframe(show_df, use_container_width=True)

csv_bytes = show_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered data (CSV)",
    data=csv_bytes,
    file_name="predictions_filtered.csv",
    mime="text/csv"
)

# =========================================================
# 7. About
# =========================================================
with st.expander("About this app"):
    st.markdown("""
**Restaurant Rating Predictor**

- Loads your dataset from Google Drive on each run (cached during session).
- Trains:
  - A regression model (best of DT / RF / GB) to predict Google rating.
  - A stacking classifier (RF + GB → LogisticRegression) to estimate probability of being "highly rated" (>4.5).
- Interactive filters and beautiful map (pydeck Scatterplot + optional HexagonLayer).

_You can cache models to avoid re-training every rerun. Here it's done via `@st.cache_resource`._
""")

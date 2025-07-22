# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests
from io import BytesIO

# ===================== Page Config =====================
st.set_page_config(page_title="Restaurant Rating Predictor", page_icon="🍽️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {background-color: #f5f5f7;}
</style>
""", unsafe_allow_html=True)

# ===================== Data URL =====================
DATA_URL = "https://drive.google.com/uc?export=download&id=1iRWeGaDybybQ2eiTCyEgyXDYbH5FpFup"

# ===================== Load Data =====================
@st.cache_data(show_spinner="Downloading data...")
def load_raw(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return pd.read_csv(BytesIO(r.content))

try:
    df_raw = load_raw(DATA_URL)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# Optional: if you ALREADY computed predictions offline, just use them and skip training.
# ---------- OFFLINE MODE EXAMPLE ----------
# if {"predicted_rating", "probability"}.issubset(df_raw.columns):
#     st.session_state["df_pred"] = df_raw  # so we can reuse the same code below
#     st.success("Loaded precomputed predictions from CSV.")
# -----------------------------------------

# ===================== Model Training Utils =====================
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier

def prepare_data(df: pd.DataFrame):
    df = df.copy()

    if "postal_code" not in df.columns and "Geographic Area Name" in df.columns:
        df["postal_code"] = df["Geographic Area Name"].astype(str).str.extract(r"(\d{5})")

    # Targets
    if "google_rating" not in df.columns:
        raise ValueError("Column 'google_rating' is required.")
    df["highly_rated"] = (df["google_rating"] > 4.5).astype(int)

    required = [
        "price","popularity","sentiment","num_reviews","category","google_rating",
        "mean_income","unemployment_percent","poverty_percent","latitude","longitude"
    ]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # Bin reviews
    q1, q2 = df["num_reviews"].quantile([0.33, 0.66])
    df["reviews_bin"] = pd.cut(
        df["num_reviews"], bins=[-1, q1, q2, df["num_reviews"].max()],
        labels=["low","medium","high"]
    )

    # Feature sets
    num_feats_reg = ["price","popularity","sentiment"]
    cat_feats_reg = ["reviews_bin","category"]
    X_reg = df[num_feats_reg + cat_feats_reg]
    y_reg = df["google_rating"]

    num_feats_clf = ["mean_income","unemployment_percent","poverty_percent",
                     "price","popularity","sentiment"]
    cat_feats_clf = ["reviews_bin","category"]
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

    return df, (X_reg, y_reg, prep_reg), (X_clf, y_clf, prep_clf)

def train_models(df: pd.DataFrame):
    """Train (lighter) models and return df with predictions."""
    df, reg_pack, clf_pack = prepare_data(df)
    X_reg, y_reg, prep_reg = reg_pack
    X_clf, y_clf, prep_clf = clf_pack

    # --- Regression: small grid to speed up ---
    models_reg = {
        "RandomForest": (
            RandomForestRegressor(random_state=42, n_jobs=1),
            {"reg__n_estimators":[150], "reg__max_depth":[None, 15]}
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {"reg__n_estimators":[200], "reg__learning_rate":[0.1]}
        )
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    best_rmse = 1e9
    best_name, best_params = None, None

    for name, (est, params) in models_reg.items():
        pipe = Pipeline([("prep", prep_reg), ("reg", est)])
        gs = GridSearchCV(pipe, params, cv=kf, scoring="neg_mean_squared_error", n_jobs=1)
        gs.fit(X_reg, y_reg)
        mse = mean_squared_error(y_reg, gs.predict(X_reg))
        rmse = mse ** 0.5
        if rmse < best_rmse:
            best_rmse, best_name, best_params = rmse, name, gs.best_params_

    reg_model = models_reg[best_name][0]
    reg_pipe = Pipeline([("prep", prep_reg), ("reg", reg_model)])
    reg_pipe.set_params(**best_params)
    reg_pipe.fit(X_reg, y_reg)

    # --- Classification: no stacking to save time ---
    gb_pipe = Pipeline([
        ("prep", prep_clf),
        ("clf", GradientBoostingClassifier(random_state=42))
    ])
    gb_pipe.fit(X_clf, y_clf)

    df["predicted_rating"] = reg_pipe.predict(X_reg).round(1)
    df["probability"] = gb_pipe.predict_proba(X_clf)[:,1]

    return df, best_rmse, best_name

# ===================== Sidebar =====================
st.sidebar.title("Controls")

# Train button (only if not trained yet)
if "df_pred" not in st.session_state:
    if st.sidebar.button("🚀 Train models now"):
        with st.spinner("Training models... (first time only)"):
            try:
                pred_df, rmse, model_name = train_models(df_raw)
                st.session_state["df_pred"] = pred_df
                st.session_state["rmse"] = rmse
                st.session_state["model_name"] = model_name
                st.success("Training finished.")
            except Exception as e:
                st.error(f"Training failed: {e}")
else:
    st.sidebar.write("Models already trained (cached).")
    if st.sidebar.button("🔁 Retrain (clear cache)"):
        for k in ["df_pred", "rmse", "model_name"]:
            st.session_state.pop(k, None)
        st.experimental_rerun()

# Use predicted or actual rating for color
color_choice = st.sidebar.radio("Color points by", ["Predicted rating", "Actual rating"])
prob_threshold = st.sidebar.slider("Min probability", 0.0, 1.0, 0.0, 0.01)

# ===================== Decide which DF to show =====================
if "df_pred" in st.session_state:
    df = st.session_state["df_pred"]
    rmse = st.session_state["rmse"]
    model_name = st.session_state["model_name"]
else:
    # No model yet -> just use raw df (no preds)
    df = df_raw.copy()
    rmse = None
    model_name = None
    if "predicted_rating" not in df.columns:
        df["predicted_rating"] = np.nan
    if "probability" not in df.columns:
        df["probability"] = 0.0

# ===================== Filters =====================
if "category" in df.columns:
    cats = sorted(df["category"].dropna().astype(str).unique())
    selected_cats = st.sidebar.multiselect("Category filter", cats)
else:
    selected_cats = []

if "price" in df.columns and pd.api.types.is_numeric_dtype(df["price"]):
    pmin, pmax = int(df["price"].min()), int(df["price"].max())
    price_range = st.sidebar.slider("Price range", pmin, pmax, (pmin, pmax))
else:
    price_range = None

mask = pd.Series(True, index=df.index)
if selected_cats and "category" in df.columns:
    mask &= df["category"].astype(str).isin(selected_cats)
if price_range and "price" in df.columns:
    mask &= df["price"].between(*price_range)
if "probability" in df.columns:
    mask &= df["probability"] >= prob_threshold

show_df = df.loc[mask].copy()

# ===================== KPIs =====================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Records", len(show_df))
if "google_rating" in show_df.columns and pd.api.types.is_numeric_dtype(show_df["google_rating"]):
    col2.metric("Avg actual rating", f"{show_df['google_rating'].mean():.2f}")
else:
    col2.metric("Avg actual rating", "N/A")
col3.metric("Avg predicted rating", f"{show_df['predicted_rating'].mean():.2f}")
col4.metric("Avg probability", f"{show_df['probability'].mean():.2f}")

if rmse is not None:
    st.caption(f"Regressor: **{model_name}**, RMSE ≈ **{rmse:.3f}** (on full data).")

# ===================== Map =====================
st.subheader("Interactive Map")

if {"latitude","longitude"}.issubset(show_df.columns):
    color_source = show_df["predicted_rating"] if color_choice == "Predicted rating" else show_df.get("google_rating", show_df["predicted_rating"])
    rmin, rmax = color_source.min(), color_source.max()

    def rating_to_rgba(r):
        if pd.isna(r) or rmax == rmin:
            return [180,180,180,170]
        t = (r - rmin) / (rmax - rmin)
        return [int(255*(1-t)), int(120+135*t), int(200*(1-t)), 180]

    show_df["color"] = color_source.apply(rating_to_rgba)

    view = pdk.ViewState(
        latitude=float(show_df["latitude"].mean()),
        longitude=float(show_df["longitude"].mean()),
        zoom=11
    )

    tooltip = "{name}\\nPred: {predicted_rating}\\nProb: {probability:.2f}"
    if "google_rating" in show_df.columns:
        tooltip += "\\nActual: {google_rating}"
    if "category" in show_df.columns:
        tooltip += "\\nCat: {category}"

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=show_df,
        get_position='[longitude, latitude]',
        get_radius=60,
        get_fill_color="color",
        pickable=True
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

    st.pydeck_chart(pdk.Deck(initial_view_state=view, layers=layers, tooltip={"text": tooltip}))
else:
    st.error("No latitude/longitude columns.")

# ===================== Table & Download =====================
with st.expander("Show data table"):
    st.dataframe(show_df, use_container_width=True)

st.download_button(
    "Download filtered CSV",
    data=show_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_predictions.csv",
    mime="text/csv"
)

# ===================== About =====================
with st.expander("About"):
    st.markdown("""
**How this app works**

- Data is pulled from Google Drive each run and cached.
- Heavy model training is run **only when you click the button** (cached afterwards).
- You can precompute predictions offline and skip in-app training for faster startup.
""")

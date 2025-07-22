# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests
from io import BytesIO

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Restaurant Rating Predictor", page_icon="🍽️", layout="wide")

# Tiny CSS for nicer look
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {background-color: #f5f5f7;}
.card-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px;}
.card {background: #fff; border-radius: 12px; padding: 16px 18px; box-shadow: 0 4px 14px rgba(0,0,0,.08);}
.card h4 {margin: 0 0 8px 0;}
.badge {display:inline-block; padding:2px 8px; border-radius:6px; font-size:12px; margin-left:6px; background:#ffb300; color:#fff;}
.prog-wrap {background:#e0e0e0; border-radius:6px; height:10px; overflow:hidden; margin-top:6px;}
.prog-bar {background:#43a047; height:100%; transition:width .4s;}
</style>
""", unsafe_allow_html=True)

# Google Drive direct link (you chose way B)
DATA_URL = "https://drive.google.com/uc?export=download&id=1iRWeGaDybybQ2eiTCyEgyXDYbH5FpFup"

# Toggle: if True, DON'T auto load/train at startup (keeps health check safe)
SAFE_START = True

# ---------------------- LOAD RAW DATA ----------------------
@st.cache_data(show_spinner="Downloading data...")
def load_raw(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return pd.read_csv(BytesIO(r.content))

# If safe start, wait for user click to load data
if SAFE_START and "df_raw" not in st.session_state:
    st.title("Restaurant Rating Predictor")
    st.info("Click the button below to load data (and then train/predict).")
    if st.button("Load data"):
        with st.spinner("Loading..."):
            st.session_state["df_raw"] = load_raw(DATA_URL)
        st.experimental_rerun()
    st.stop()

# Data is ready here
df_raw = st.session_state.get("df_raw", load_raw(DATA_URL))

# ---------------------- ML UTILS ----------------------
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier

def prepare_data(df: pd.DataFrame):
    df = df.copy()
    if "postal_code" not in df.columns and "Geographic Area Name" in df.columns:
        df["postal_code"] = df["Geographic Area Name"].astype(str).str.extract(r"(\d{5})")
    if "google_rating" not in df.columns:
        raise ValueError("Column 'google_rating' is required to train.")

    df["highly_rated"] = (df["google_rating"] > 4.5).astype(int)

    req = ["price","popularity","sentiment","num_reviews","category","google_rating",
           "mean_income","unemployment_percent","poverty_percent","latitude","longitude"]
    df = df.dropna(subset=[c for c in req if c in df.columns])

    q1, q2 = df["num_reviews"].quantile([0.33, 0.66])
    df["reviews_bin"] = pd.cut(df["num_reviews"], bins=[-1, q1, q2, df["num_reviews"].max()],
                               labels=["low","medium","high"])

    # Regression
    num_r = ["price","popularity","sentiment"]
    cat_r = ["reviews_bin","category"]
    X_reg = df[num_r + cat_r]; y_reg = df["google_rating"]
    prep_reg = ColumnTransformer([
        ("num", StandardScaler(), num_r),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_r)
    ])

    # Classification
    num_c = ["mean_income","unemployment_percent","poverty_percent",
             "price","popularity","sentiment"]
    cat_c = ["reviews_bin","category"]
    X_clf = df[num_c + cat_c]; y_clf = df["highly_rated"]
    prep_clf = ColumnTransformer([
        ("num", StandardScaler(), num_c),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_c)
    ])

    return df, (X_reg, y_reg, prep_reg), (X_clf, y_clf, prep_clf)

def train_models(df: pd.DataFrame):
    df, (X_reg, y_reg, prep_reg), (X_clf, y_clf, prep_clf) = prepare_data(df)

    # --- Light regression grid (fast) ---
    models_reg = {
        "RF": (RandomForestRegressor(random_state=42, n_jobs=1),
               {"reg__n_estimators":[150], "reg__max_depth":[None, 15]}),
        "GB": (GradientBoostingRegressor(random_state=42),
               {"reg__n_estimators":[200], "reg__learning_rate":[0.1]}),
        "DT": (DecisionTreeRegressor(random_state=42),
               {"reg__max_depth":[None, 8]})
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    best_rmse, best_name, best_params = 1e9, None, None

    for name, (est, params) in models_reg.items():
        pipe = Pipeline([("prep", prep_reg), ("reg", est)])
        gs = GridSearchCV(pipe, params, cv=kf, scoring="neg_mean_squared_error", n_jobs=1)
        gs.fit(X_reg, y_reg)
        rmse = (mean_squared_error(y_reg, gs.predict(X_reg)))**0.5
        if rmse < best_rmse:
            best_rmse, best_name, best_params = rmse, name, gs.best_params_

    reg_model = models_reg[best_name][0]
    reg_pipe = Pipeline([("prep", prep_reg), ("reg", reg_model)])
    reg_pipe.set_params(**best_params)
    reg_pipe.fit(X_reg, y_reg)

    # --- Classification (simplified: only GB) ---
    gb_pipe = Pipeline([("prep", prep_clf),
                        ("clf", GradientBoostingClassifier(random_state=42))])
    gb_pipe.fit(X_clf, y_clf)

    df["predicted_rating"] = reg_pipe.predict(X_reg).round(1)
    df["probability"] = gb_pipe.predict_proba(X_clf)[:,1]

    return df, best_rmse, best_name

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("Controls")

# Train button
if "df_pred" not in st.session_state:
    if st.sidebar.button("🚀 Train / Predict"):
        with st.spinner("Training models..."):
            try:
                pred_df, rmse, model_name = train_models(df_raw)
                st.session_state["df_pred"] = pred_df
                st.session_state["rmse"] = rmse
                st.session_state["model_name"] = model_name
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.stop()
        st.experimental_rerun()
else:
    if st.sidebar.button("🔁 Retrain (clear cache)"):
        for k in ["df_pred","rmse","model_name"]:
            st.session_state.pop(k, None)
        st.experimental_rerun()

# Decide which df to use
if "df_pred" in st.session_state:
    df = st.session_state["df_pred"]
    rmse = st.session_state["rmse"]
    model_name = st.session_state["model_name"]
else:
    df = df_raw.copy()
    if "predicted_rating" not in df.columns: df["predicted_rating"] = np.nan
    if "probability" not in df.columns: df["probability"] = 0.0
    rmse = model_name = None

# ----- Sidebar filters -----
# City select (single)
if "city" in df.columns:
    cities = ["All cities"] + sorted(df["city"].dropna().astype(str).unique())
    selected_city = st.sidebar.selectbox("City", cities)
else:
    selected_city = "All cities"

# Category multi-select
if "category" in df.columns:
    cats = sorted(df["category"].dropna().astype(str).unique())
    selected_cats = st.sidebar.multiselect("Category", cats)
else:
    selected_cats = []

# Price
if "price" in df.columns and pd.api.types.is_numeric_dtype(df["price"]):
    pmin, pmax = int(df["price"].min()), int(df["price"].max())
    price_range = st.sidebar.slider("Price range", pmin, pmax, (pmin, pmax))
else:
    price_range = None

# Prob threshold
prob_threshold = st.sidebar.slider("Min probability", 0.0, 1.0, 0.0, 0.01)

# Sort metric for Top 5
sort_metric = st.sidebar.radio("Top-5 sort by", ["Predicted rating", "Probability"])

# Color by
color_choice = st.sidebar.radio("Map color by", ["Predicted rating", "Actual rating"])

# ----- Apply filters -----
mask = pd.Series(True, index=df.index)
if selected_city != "All cities" and "city" in df.columns:
    mask &= df["city"].astype(str) == selected_city
if selected_cats and "category" in df.columns:
    mask &= df["category"].astype(str).isin(selected_cats)
if price_range and "price" in df.columns:
    mask &= df["price"].between(*price_range)
mask &= df["probability"] >= prob_threshold

filtered_df = df.loc[mask].copy()

# ---------------------- KPIs ----------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Records", len(filtered_df))
if "google_rating" in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df["google_rating"]):
    col2.metric("Avg actual rating", f"{filtered_df['google_rating'].mean():.2f}")
else:
    col2.metric("Avg actual rating", "N/A")
col3.metric("Avg predicted rating", f"{filtered_df['predicted_rating'].mean():.2f}")
col4.metric("Avg probability", f"{filtered_df['probability'].mean():.2f}")

if rmse is not None:
    st.caption(f"Regressor: **{model_name}**, RMSE ≈ **{rmse:.3f}**")

# ---------------------- TOP 5 SECTION ----------------------
st.subheader("Top 5 Restaurants")

if sort_metric == "Predicted rating":
    top_df = filtered_df.sort_values("predicted_rating", ascending=False).head(5)
else:
    top_df = filtered_df.sort_values("probability", ascending=False).head(5)

# Cards
def show_cards(df_top):
    html_cards = []
    for _, r in df_top.iterrows():
        prob_pct = int(r["probability"] * 100) if "probability" in r else 0
        rating_badge = f'<span class="badge">{r["predicted_rating"]}</span>' if not pd.isna(r["predicted_rating"]) else ""
        html_cards.append(f"""
        <div class="card">
            <h4>{r.get('name','(no name)')}{rating_badge}</h4>
            <div><b>City:</b> {r.get('city','-')}</div>
            <div><b>ZIP:</b> {r.get('postal_code','-')}</div>
            <div><b>Category:</b> {r.get('category','-')}</div>
            <div><b>Probability:</b> {prob_pct}%</div>
            <div class="prog-wrap"><div class="prog-bar" style="width:{prob_pct}%"></div></div>
        </div>
        """)
    st.markdown(f'<div class="card-grid">{"".join(html_cards)}</div>', unsafe_allow_html=True)

if len(top_df):
    show_cards(top_df)
else:
    st.info("No records match your filters.")

# ---------------------- MAP ----------------------
st.subheader("Map (Top 5)")

if len(top_df) and {"latitude","longitude"}.issubset(top_df.columns):
    # color by chosen metric
    if color_choice == "Actual rating" and "google_rating" in top_df.columns:
        color_source = top_df["google_rating"]
    else:
        color_source = top_df["predicted_rating"]

    rmin, rmax = color_source.min(), color_source.max()
    def rating_to_rgba(r):
        if pd.isna(r) or rmax == rmin:
            return [180,180,180,170]
        t = (r - rmin) / (rmax - rmin)
        return [int(255*(1-t)), int(120+135*t), int(200*(1-t)), 180]

    top_df = top_df.copy()
    top_df["color"] = color_source.apply(rating_to_rgba)

    view = pdk.ViewState(
        latitude=float(top_df["latitude"].mean()),
        longitude=float(top_df["longitude"].mean()),
        zoom=12
    )

    tooltip = "{name}\\nPred: {predicted_rating}\\nProb: {probability:.2f}"
    if "google_rating" in top_df.columns:
        tooltip += "\\nActual: {google_rating}"
    if "category" in top_df.columns:
        tooltip += "\\nCat: {category}"

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=top_df,
        get_position='[longitude, latitude]',
        get_radius=90,
        get_fill_color="color",
        pickable=True
    )

    st.pydeck_chart(pdk.Deck(initial_view_state=view, layers=[layer], tooltip={"text": tooltip}))
else:
    st.warning("No map data for Top 5 (missing lat/lon or no records).")

# ---------------------- FULL TABLE ----------------------
with st.expander("Show filtered data table"):
    st.dataframe(filtered_df, use_container_width=True)

st.download_button(
    "Download filtered CSV",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_predictions.csv",
    mime="text/csv"
)

# ---------------------- ABOUT ----------------------
with st.expander("About"):
    st.markdown("""
- Choose **City** then we sort & show **Top 5** by Predicted rating or Probability.
- Cards show probability bars; map only shows those Top 5 (cleaner view).
- Training runs ONLY when you click the button (cached afterwards).  
  If startup仍慢，建议离线生成 `predicted_rating` & `probability` 再上传。
""")

# app.py
import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
from io import BytesIO
from pathlib import Path

# ================== 基本设置 ==================
st.set_page_config(page_title="restaurant", layout="wide")

# 你的 Google Drive 直链（方式 B）
DATA_URL = "https://drive.google.com/uc?export=download&id=1iRWeGaDybybQ2eiTCyEgyXDYbH5FpFup"

# ================== 读数据函数 ==================
@st.cache_data(show_spinner="Downloading data from Google Drive...")
def load_data():
    """
    从 Google Drive 下载 CSV 并读入 DataFrame。
    若失败，会给出提示并返回一个示例数据，保证页面能跑起来。
    """
    try:
        r = requests.get(DATA_URL, timeout=60)
        r.raise_for_status()
        return pd.read_csv(BytesIO(r.content))
    except Exception as e:
        st.warning(f"下载或读取数据失败：{e}\n将展示示例数据。")
        return pd.DataFrame({
            "name": ["Demo A", "Demo B", "Demo C"],
            "latitude": [37.7749, 37.7849, 37.7649],
            "longitude": [-122.4194, -122.4094, -122.4294],
            "rating": [4.6, 4.2, 4.8],
            "category": ["Asian", "Cafe", "Italian"],
            "price": [2, 1, 3],
            "popularity": [0.8, 0.5, 0.9]
        })

df = load_data()

# 选一个评分列名（有的叫 google_rating，有的叫 rating）
rating_col = "google_rating" if "google_rating" in df.columns else ("rating" if "rating" in df.columns else None)

# ================== 侧边栏筛选 ==================
st.sidebar.header("筛选条件")

# 类别筛选
if "category" in df.columns:
    categories = sorted(df["category"].dropna().astype(str).unique())
    selected_cats = st.sidebar.multiselect("类别 (category)", categories)
else:
    selected_cats = []

# 价格筛选
if "price" in df.columns and pd.api.types.is_numeric_dtype(df["price"]):
    pmin, pmax = int(df["price"].min()), int(df["price"].max())
    price_range = st.sidebar.slider("价格区间 (price)", pmin, pmax, (pmin, pmax))
else:
    price_range = None

# 人气筛选（可选）
if "popularity" in df.columns and pd.api.types.is_numeric_dtype(df["popularity"]):
    pop_min, pop_max = float(df["popularity"].min()), float(df["popularity"].max())
    popularity_range = st.sidebar.slider("人气区间 (popularity)", pop_min, pop_max, (pop_min, pop_max))
else:
    popularity_range = None

# 过滤
mask = pd.Series([True] * len(df))
if selected_cats and "category" in df.columns:
    mask &= df["category"].astype(str).isin(selected_cats)
if price_range and "price" in df.columns:
    mask &= df["price"].between(*price_range)
if popularity_range and "popularity" in df.columns:
    mask &= df["popularity"].between(*popularity_range)

show_df = df[mask].copy()

st.write(f"当前显示：**{len(show_df)}** 条记录")

# ================== 地图 ==================
if {"latitude", "longitude"}.issubset(show_df.columns):
    lat_mean = show_df["latitude"].mean()
    lon_mean = show_df["longitude"].mean()

    initial_view = pdk.ViewState(
        latitude=lat_mean,
        longitude=lon_mean,
        zoom=11
    )

    tooltip_text = "{name}"
    if rating_col and rating_col in show_df.columns:
        tooltip_text += f"\n评分: {{{rating_col}}}"
    if "category" in show_df.columns:
        tooltip_text += "\n类别: {category}"

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=show_df,
        get_position='[longitude, latitude]',
        get_radius=60,
        get_fill_color=[255, 0, 0, 140],
        pickable=True,
    )

    st.pydeck_chart(pdk.Deck(
        initial_view_state=initial_view,
        layers=[layer],
        tooltip={"text": tooltip_text}
    ))
else:
    st.error("数据里找不到 latitude / longitude 列，无法绘制地图。")

# ================== 数据表 ==================
with st.expander("查看数据表"):
    st.dataframe(show_df)

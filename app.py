import streamlit as st
import pandas as pd
import pydeck as pdk
from pathlib import Path

st.set_page_config(page_title="餐厅地图", layout="wide")

# ---------- 1. 读数据 ----------
@st.cache_data
def load_data():
    csv_path = Path("data/enriched_new.csv")  # 你之后上传的文件名
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        # 没有数据时，用一个小示例让页面能跑起来
        df = pd.DataFrame({
            "name": ["Demo A", "Demo B", "Demo C"],
            "latitude": [37.7749, 37.7849, 37.7649],
            "longitude": [-122.4194, -122.4094, -122.4294],
            "rating": [4.6, 4.2, 4.8],
            "category": ["Asian", "Cafe", "Italian"],
            "price": [2, 1, 3],
            "popularity": [0.8, 0.5, 0.9]
        })
    return df

df = load_data()

# ---------- 2. 侧边栏筛选 ----------
st.sidebar.header("筛选条件")
cats = st.sidebar.multiselect("类别 (category)", sorted(df["category"].dropna().unique()) if "category" in df else [])
price_min, price_max = (int(df["price"].min()), int(df["price"].max())) if "price" in df else (0, 5)
selected_price = st.sidebar.slider("价格区间 (price)", price_min, price_max, (price_min, price_max))

mask = pd.Series([True] * len(df))
if cats:
    mask &= df["category"].isin(cats)
if "price" in df:
    mask &= df["price"].between(*selected_price)

show_df = df[mask].copy()

st.write(f"当前显示：**{len(show_df)}** 家餐厅")

# ---------- 3. 地图 ----------
if "latitude" in show_df and "longitude" in show_df:
    initial_view = pdk.ViewState(
        latitude=show_df["latitude"].mean(),
        longitude=show_df["longitude"].mean(),
        zoom=11
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=show_df,
        get_position='[longitude, latitude]',
        get_radius=60,
        get_fill_color=[255, 0, 0, 140],
        pickable=True,
    )

    tooltip = {"text": "{name}\n评分: {rating}\n类别: {category}"}

    st.pydeck_chart(pdk.Deck(initial_view_state=initial_view, layers=[layer], tooltip=tooltip))
else:
    st.error("数据里找不到 latitude/longitude 列")

# ---------- 4. 表格查看 ----------
with st.expander("查看数据表"):
    st.dataframe(show_df)

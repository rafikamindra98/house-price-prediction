from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Ames House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent

@st.cache_resource
def load_resources():
    bundle = joblib.load(ROOT / "house_price_artifacts" / "house_price_model.joblib")
    with open(ROOT / "house_price_artifacts" / "feature_defaults.json", "r") as f:
        defaults = json.load(f)
    with open(ROOT / "house_price_artifacts" / "metadata.json", "r") as f:
        metadata = json.load(f)
    return bundle, defaults, metadata

bundle, defaults, metadata = load_resources()

# Header
st.title("🏠 Ames House Price Predictor")
st.markdown(
    "Aplikasi estimasi nilai wajar properti perumahan berbasis **Ensemble Machine Learning (Ridge Regression 60% + XGBoost 40%)** "
    "menggunakan dataset Ames Housing (*Kaggle: Advanced Regression Techniques*)."
)

# Metric Badges
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Test R² Score", f"{metadata['test_r2'] * 100:.2f}%")
with col_m2:
    st.metric("Test Log-RMSE", f"{metadata['test_log_rmse']:.4f}")
with col_m3:
    st.metric("Mean Absolute Error (MAE)", f"${metadata['test_mae_usd']:,.2f}")
with col_m4:
    st.metric("Arsitektur Model", "Ridge (60%) + XGBoost (40%)")

st.markdown("---")

st.subheader("📋 Masukkan Spesifikasi Properti")

# Form input organized in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 📍 Lokasi & Kualitas Umum")
    neighborhood_options = bundle["category_options"].get("Neighborhood", ["CollgCr", "NAmes", "Edwards"])
    neighborhood = st.selectbox(
        "Kawasan / Neighborhood",
        options=neighborhood_options,
        index=neighborhood_options.index("CollgCr") if "CollgCr" in neighborhood_options else 0
    )
    
    overall_qual = st.slider(
        "Kualitas Material & Finishing (OverallQual)",
        min_value=1, max_value=10, value=7,
        help="1 = Sangat Buruk, 5 = Rata-rata, 10 = Sangat Mewah"
    )
    
    overall_cond = st.slider(
        "Kondisi Fisik Rumah (OverallCond)",
        min_value=1, max_value=10, value=5,
        help="Skala 1 - 10"
    )
    
    year_built = st.number_input("Tahun Dibangun (YearBuilt)", min_value=1870, max_value=2026, value=2003, step=1)
    year_remod = st.number_input("Tahun Renovasi (YearRemodAdd)", min_value=1950, max_value=2026, value=2004, step=1)
    yr_sold = st.number_input("Tahun Penjualan / Transaksi", min_value=2006, max_value=2026, value=2010, step=1)

with col2:
    st.markdown("##### 📐 Dimensi & Luas Lantai")
    gr_liv_area = st.number_input(
        "Luas Lantai Atas Tanah (GrLivArea, sq ft)",
        min_value=300, max_value=6000, value=1700, step=25,
        help="Luas ruang huni lantai dasar dan atas (tidak termasuk basement)"
    )
    
    total_bsmt_sf = st.number_input(
        "Luas Basement (TotalBsmtSF, sq ft)",
        min_value=0, max_value=4000, value=900, step=25
    )
    
    lot_area = st.number_input(
        "Luas Lahan / Tanah (LotArea, sq ft)",
        min_value=1000, max_value=100000, value=9500, step=100
    )
    
    flr1_sf = st.number_input("Luas Lantai 1 (1stFlrSF, sq ft)", min_value=300, max_value=4000, value=1000, step=25)
    flr2_sf = st.number_input("Luas Lantai 2 (2ndFlrSF, sq ft)", min_value=0, max_value=3000, value=700, step=25)

with col3:
    st.markdown("##### 🛁 Ruangan & Fasilitas")
    bedroom_abv_gr = st.number_input("Kamar Tidur di Atas Tanah", min_value=1, max_value=8, value=3, step=1)
    full_bath = st.number_input("Kamar Mandi Penuh (FullBath)", min_value=0, max_value=5, value=2, step=1)
    half_bath = st.number_input("Toilet Setengah (HalfBath)", min_value=0, max_value=4, value=1, step=1)
    
    garage_cars = st.slider("Kapasitas Garasi (Jumlah Mobil)", min_value=0, max_value=4, value=2)
    garage_area = st.number_input("Luas Garasi (GarageArea, sq ft)", min_value=0, max_value=1500, value=500, step=25)
    
    kitchen_qual_options = ["Ex", "Gd", "TA", "Fa"]
    kitchen_qual = st.selectbox(
        "Kualitas Dapur (KitchenQual)",
        options=kitchen_qual_options,
        index=1,
        format_func=lambda x: {"Ex": "Excellent (Mewah)", "Gd": "Good (Bagus)", "TA": "Typical/Average (Standar)", "Fa": "Fair (Cukup)"}.get(x, x)
    )
    
    central_air = st.selectbox("AC Sentral (CentralAir)", options=["Y", "N"], index=0)
    fireplaces = st.number_input("Jumlah Perapian (Fireplaces)", min_value=0, max_value=4, value=1, step=1)

# Prediction Button
if st.button("🚀 Hitung Estimasi Harga Rumah", type="primary", use_container_width=True):
    # Construct complete input dictionary
    row = dict(defaults)
    
    # Override with user inputs
    row["Neighborhood"] = neighborhood
    row["OverallQual"] = overall_qual
    row["OverallCond"] = overall_cond
    row["YearBuilt"] = year_built
    row["YearRemodAdd"] = year_remod
    row["YrSold"] = yr_sold
    row["GrLivArea"] = gr_liv_area
    row["TotalBsmtSF"] = total_bsmt_sf
    row["LotArea"] = lot_area
    row["1stFlrSF"] = flr1_sf
    row["2ndFlrSF"] = flr2_sf
    row["BedroomAbvGr"] = bedroom_abv_gr
    row["FullBath"] = full_bath
    row["HalfBath"] = half_bath
    row["GarageCars"] = garage_cars
    row["GarageArea"] = garage_area
    row["KitchenQual"] = kitchen_qual
    row["CentralAir"] = central_air
    row["Fireplaces"] = fireplaces
    
    # Feature Engineering
    row["TotalSF"] = float(total_bsmt_sf) + float(flr1_sf) + float(flr2_sf)
    row["TotalBath"] = float(full_bath) + 0.5 * float(half_bath) + float(row.get("BsmtFullBath", 0)) + 0.5 * float(row.get("BsmtHalfBath", 0))
    row["HouseAge"] = max(0, int(yr_sold) - int(year_built))
    row["RemodAge"] = max(0, int(yr_sold) - int(year_remod))
    row["HasPool"] = int(row.get("PoolArea", 0) > 0)
    row["HasGarage"] = int(garage_area > 0)
    row["HasBasement"] = int(total_bsmt_sf > 0)
    row["HasFireplace"] = int(fireplaces > 0)
    
    # Convert to DataFrame with exact column order
    input_df = pd.DataFrame([row])
    
    # Reorder columns to match pipeline feature_names
    input_df = input_df[bundle["feature_names"]]
    
    # Predict with Ridge and XGBoost
    pred_ridge_log = float(bundle["ridge_model"].predict(input_df)[0])
    pred_xgb_log = float(bundle["xgb_model"].predict(input_df)[0])
    
    # Weighted Blend (60% Ridge + 40% XGBoost)
    pred_blend_log = 0.60 * pred_ridge_log + 0.40 * pred_xgb_log
    
    # Inverse transform
    pred_usd = float(np.expm1(pred_blend_log))
    ridge_usd = float(np.expm1(pred_ridge_log))
    xgb_usd = float(np.expm1(pred_xgb_log))
    
    # Confidence range based on Test Log-RMSE (0.1169)
    rmse = metadata["test_log_rmse"]
    lower_usd = float(np.expm1(pred_blend_log - rmse))
    upper_usd = float(np.expm1(pred_blend_log + rmse))
    
    st.markdown("---")
    st.success("### 🎉 Hasil Estimasi Nilai Pasar Properti")
    
    res_col1, res_col2 = st.columns([1.5, 1])
    
    with res_col1:
        st.metric(
            label="Estimasi Nilai Pasar (USD)",
            value=f"${pred_usd:,.0f}",
            help="Hasil pembobotan ensemble 60% Ridge + 40% XGBoost"
        )
        st.info(f"**Rentang Wajar (Confidence Interval 68%):** **${lower_usd:,.0f}** — **${upper_usd:,.0f}**")
        st.caption(f"Margin error diperhitungkan dari Log-RMSE uji sebesar {rmse:.4f} (~11.7% variasi wajar).")
    
    with res_col2:
        st.markdown("##### 🔍 Perbandingan Prediksi Model")
        comp_df = pd.DataFrame({
            "Arsitektur Model": ["Ridge Regression (60%)", "XGBoost Regressor (40%)", "Ensemble Blended (Final)"],
            "Prediksi Harga": [f"${ridge_usd:,.0f}", f"${xgb_usd:,.0f}", f"${pred_usd:,.0f}"]
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

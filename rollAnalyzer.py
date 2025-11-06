import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Roll Analyzer", layout="wide")
st.title("SR3 Roll Behavior Analyzer")

# File Upload
data_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
if data_file is None:
    st.stop()

# Load data
df = pd.read_excel(data_file)

# Debug: Show raw data
with st.expander("📋 Debug: Raw Data Preview"):
    st.write("First few rows of raw data:")
    st.dataframe(df.head())
    st.write(f"First column name: '{df.columns[0]}'")
    st.write(f"First column data type: {df[df.columns[0]].dtype}")
    st.write(f"Sample values from first column: {df[df.columns[0]].head(3).tolist()}")

# Identify timestamp column (first column)
ts_col = df.columns[0]

# Try multiple date parsing strategies
if df[ts_col].dtype == 'object':
    # Try parsing as string with various formats
    df[ts_col] = pd.to_datetime(df[ts_col], format='%m/%d/%Y', errors='coerce')
    if df[ts_col].isna().all():
        df[ts_col] = pd.to_datetime(df[ts_col], format='%d/%m/%Y', errors='coerce')
    if df[ts_col].isna().all():
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
else:
    # Already datetime or numeric
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

# Set index
df = df.set_index(ts_col)

# Remove any NaT rows
original_rows = len(df)
df = df[df.index.notna()]
rows_after = len(df)

if rows_after == 0:
    st.error(f"❌ All {original_rows} rows have invalid dates!")
    st.error("Please check that your Timestamp column contains valid dates.")
    st.stop()
elif rows_after < original_rows:
    st.warning(f"⚠️ Removed {original_rows - rows_after} rows with invalid dates")

st.success(f"✅ Loaded {rows_after} rows with valid dates from {df.index.min().date()} to {df.index.max().date()}")

# Extract contract names from column headers
# Look for patterns like: SRAU25_Price, SRAU25 Price, or just the contract codes
contracts = []
for col in df.columns:
    col_str = str(col)
    # Try pattern 1: CONTRACT_Price or CONTRACT_OI
    if '_Price' in col_str or '_OI' in col_str:
        contract = col_str.split('_')[0]
        if contract not in contracts:
            contracts.append(contract)
    # Try pattern 2: CONTRACT Price or CONTRACT OI (with space)
    elif ' Price' in col_str or ' OI' in col_str:
        contract = col_str.split(' ')[0]
        if contract not in contracts:
            contracts.append(contract)

contracts = sorted(contracts)

if len(contracts) == 0:
    st.error("❌ No contracts found. Your columns should be named like: SRAU25_Price and SRAU25_OI")
    st.info("💡 Or try: 'SRAU25 Price' and 'SRAU25 OI' (with space instead of underscore)")
    st.stop()
    
st.success(f"✅ Found {len(contracts)} contracts: {', '.join(contracts)}")

st.sidebar.header("Select Roll Pair")
front = st.sidebar.selectbox("Front Contract", contracts, index=0 if len(contracts) > 0 else None)
back = st.sidebar.selectbox("Back Contract", contracts, index=1 if len(contracts) > 1 else 0)

# Date selection with proper conversion
if len(df.index) > 0:
    min_date = df.index.min().date()
    max_date = df.index.max().date()
else:
    min_date = datetime.now().date()
    max_date = datetime.now().date()

start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

# Convert to datetime for filtering
start = pd.Timestamp(start_date)
end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # Include end date

df_win = df.loc[start:end].copy()

st.info(f"📅 Selected date range: {start_date} to {end_date} ({len(df_win)} rows)")

# Extract series
try:
    # Try underscore pattern first
    price_col_front = f"{front}_Price"
    oi_col_front = f"{front}_OI"
    price_col_back = f"{back}_Price"
    oi_col_back = f"{back}_OI"
    
    # Check if columns exist, if not try space pattern
    if price_col_front not in df_win.columns:
        price_col_front = f"{front} Price"
        oi_col_front = f"{front} OI"
        price_col_back = f"{back} Price"
        oi_col_back = f"{back} OI"
    
    price_front = df_win[price_col_front]
    price_back = df_win[price_col_back]
    oi_front = df_win[oi_col_front]
    oi_back = df_win[oi_col_back]
    
    # Show what we found
    st.info(f"📊 Using columns: {price_col_front}, {oi_col_front}, {price_col_back}, {oi_col_back}")
    
except KeyError as e:
    st.error(f"❌ Column not found: {e}")
    st.error(f"Looking for: {front}_Price, {front}_OI, {back}_Price, {back}_OI")
    st.error(f"Or: {front} Price, {front} OI, {back} Price, {back} OI")
    st.info("Available columns: " + ", ".join(df_win.columns))
    st.stop()

# Classification functions
def classify_roll(delta_front, delta_back):
    """Classify roll based on OI changes"""
    if pd.isna(delta_front) or pd.isna(delta_back):
        return "NO_DATA"
    if delta_front < 0 and delta_back > 0:
        return "MECHANICAL"
    if delta_front < 0 and delta_back < 0:
        return "EXIT"
    if delta_front > 0 and delta_back > 0:
        return "BUILD"
    if delta_front > 0 and delta_back < 0:
        return "CURVE"
    return "NEUTRAL"

def classify_sentiment(dp_f, dp_b):
    """Classify sentiment based on price changes"""
    if pd.isna(dp_f) or pd.isna(dp_b):
        return "NO_DATA"
    if dp_f > 0 and dp_b > 0:
        return "BULL"
    if dp_f < 0 and dp_b < 0:
        return "BEAR"
    if abs(dp_f) < 0.001 and abs(dp_b) < 0.001:
        return "FLAT"
    return "RV"

# Build output dataframe
out = pd.DataFrame(index=df_win.index)

# Your data already contains the CHANGES (deltas), not absolute values
# So we use them directly
out["ΔOI_Front"] = oi_front
out["ΔOI_Back"] = oi_back
out["ΔPrice_Front"] = price_front
out["ΔPrice_Back"] = price_back

# Debug: Show data summary
with st.expander("🔍 Debug: Data Summary"):
    st.write(f"Date range: {df_win.index.min()} to {df_win.index.max()}")
    st.write(f"Number of rows: {len(df_win)}")
    st.write(f"Front contract ({front}) sample data:")
    st.write(f"  - ΔPrice: {price_front.head(3).tolist()}")
    st.write(f"  - ΔOI: {oi_front.head(3).tolist()}")
    st.write(f"Back contract ({back}) sample data:")
    st.write(f"  - ΔPrice: {price_back.head(3).tolist()}")
    st.write(f"  - ΔOI: {oi_back.head(3).tolist()}")

# Apply classifications
out["Roll_Type"] = out.apply(lambda row: classify_roll(row["ΔOI_Front"], row["ΔOI_Back"]), axis=1)
out["Sentiment"] = out.apply(lambda row: classify_sentiment(row["ΔPrice_Front"], row["ΔPrice_Back"]), axis=1)

# Calculate roll spread (difference in price changes)
out["Roll_Spread"] = price_front - price_back

st.subheader(f"Roll Classification: {front} → {back}")
st.dataframe(out.style.format({
    "ΔOI_Front": "{:.0f}",
    "ΔOI_Back": "{:.0f}",
    "ΔPrice_Front": "{:.4f}",
    "ΔPrice_Back": "{:.4f}",
    "Roll_Spread": "{:.4f}"
}))

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Roll Spread Over Time")
    st.line_chart(out["Roll_Spread"])

with col2:
    st.subheader("OI Changes")
    oi_chart = pd.DataFrame({
        "Front OI": out["ΔOI_Front"],
        "Back OI": out["ΔOI_Back"]
    })
    st.line_chart(oi_chart)

# Summary statistics
summary = out.groupby(["Roll_Type", "Sentiment"]).size().reset_index(name="Days")
st.subheader("Roll Behavior Summary")
st.dataframe(summary)

# Key insights
st.subheader("Key Statistics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    build_days = len(out[out["Roll_Type"] == "BUILD"])
    st.metric("BUILD Days", build_days)
with col2:
    exit_days = len(out[out["Roll_Type"] == "EXIT"])
    st.metric("EXIT Days", exit_days)
with col3:
    mech_days = len(out[out["Roll_Type"] == "MECHANICAL"])
    st.metric("MECHANICAL Days", mech_days)
with col4:
    curve_days = len(out[out["Roll_Type"] == "CURVE"])
    st.metric("CURVE Days", curve_days)

st.markdown("""
### Interpretation Guide
- **BUILD + BULL** → New longs added → **Bullish directional roll**
- **BUILD + BEAR** → New shorts added → **Bearish directional roll**
- **BUILD + RV** → Relative value accumulation
- **MECHANICAL** → Pure expiry roll → neutral directional signal
- **EXIT + BULL/BEAR** → Profit taking or stop loss
- **CURVE** → Slope trade (steepener/flattening)

### Roll Type Definitions
- **MECHANICAL**: ΔOI(Front)↓ and ΔOI(Back)↑ → Positions moving forward
- **EXIT**: ΔOI(Front)↓ and ΔOI(Back)↓ → Traders exiting exposure
- **BUILD**: ΔOI(Front)↑ and ΔOI(Back)↑ → New risk being added ⚠️
- **CURVE**: ΔOI(Front)↑ and ΔOI(Back)↓ → Curve positioning
""")
import io
import hashlib
import streamlit as st
import pandas as pd

from data_io import load_df, to_csv_bytes

st.set_page_config(page_title="Data Zen", layout="wide")

# ---------- Helpers ----------

def _file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

@st.cache_data(show_spinner=False)
def _cached_load_df(file_bytes: bytes):
    # cache by file hash to avoid re-parsing on reruns
    return load_df(file_bytes)

def _missing_table(df: pd.DataFrame) -> pd.DataFrame:
    mc = df.isna().sum()
    pct = (mc / len(df) * 100).round(2) if len(df) else mc.astype(float)
    return pd.DataFrame({"missing_count": mc, "missing_pct": pct}).sort_values(
    "missing_pct", ascending=False)

# ---------- UI ----------

st.title("Data Zen")

uploaded = st.file_uploader("Upload a CSV file", type=["csv", "txt"])
if not uploaded:
    st.info("Upload a CSV or TXT file to preview its contents and download it back.")
    st.stop()

file_bytes = uploaded.getvalue()

try:
    df = _cached_load_df(file_bytes)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.success(f"Loaded {uploaded.name} · {df.shape[0]} rows × {df.shape[1]} columns")

# Quick profile

with st.expander("Dataset overview", expanded=True):
    c1, c2 = st.columns([2, 1])
with c1:
    st.dataframe(df.head(50), use_container_width=True)
with c2:
    st.markdown("Columns:")
    st.write(list(df.columns))
    st.markdown("Dtypes:")
    st.write(df.dtypes.astype(str))
    st.markdown("Missing values:")
    st.dataframe(_missing_table(df), use_container_width=True)

# Download button

st.download_button(label="⬇️ Download CSV",data=to_csv_bytes(df),file_name="data.csv",mime="text/csv",)

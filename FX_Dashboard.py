import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="FX Dashboard", layout="wide")

st.title("FX Strategy Dashboard")
st.write("Interactive rolling correlations and FX cross-return matrix (all rates in USD/Foreign convention).")

# --------------------------
# Helper: Standardize to USD/Foreign
# --------------------------
def standardize_to_usd_base(series, ticker):
    """
    Convert to USD/Foreign:
      - 'EURUSD=X' (EUR/USD) → invert → USD/EUR
      - 'JPY=X' (USD/JPY) → keep → USD/JPY
    """
    # Common Foreign/USD pairs on Yahoo Finance
    foreign_per_usd_tickers = {
        "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X",
        "CADUSD=X", "CHFUSD=X"  # though CAD & CHF are often quoted as USD/CAD etc., better safe
    }
    if ticker in foreign_per_usd_tickers:
        return 1 / series
    else:
        return series

# --------------------------
# User Inputs
# --------------------------
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    ticker1 = st.text_input("FX Ticker 1", value="EURUSD=X")

with col2:
    ticker2 = st.text_input("FX Ticker 2", value="GBPUSD=X")

with col3:
    window = st.number_input("Rolling Window (days)", value=30, min_value=5, max_value=365)

# --------------------------
# Download & Standardize User Tickers
# --------------------------
@st.cache_data
def get_fx(ticker, period="10y"):
    data = yf.download(ticker, period=period, auto_adjust=True)["Close"]
    if data.empty:
        raise ValueError(f"No data for {ticker}")
    return data

try:
    s1_raw = get_fx(ticker1)
    s2_raw = get_fx(ticker2)
except Exception as e:
    st.error(f"Error downloading FX data: {e}")
    st.stop()

# Standardize both to USD/Foreign
s1 = standardize_to_usd_base(s1_raw, ticker1)
s2 = standardize_to_usd_base(s2_raw, ticker2)

df = pd.concat([s1, s2], axis=1)
df.columns = [ticker1, ticker2]
df = df.dropna()

if df.empty:
    st.error("No overlapping data after standardization.")
    st.stop()

# --------------------------
# Rolling Correlation (Now Consistent!)
# --------------------------
st.subheader(f"📉 Rolling {window}-Day Correlation: {ticker1} vs {ticker2} (USD/Foreign)")

returns1 = df[ticker1].pct_change()
returns2 = df[ticker2].pct_change()

rolling_corr = returns1.rolling(window).corr(returns2)

fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(
    x=rolling_corr.index,
    y=rolling_corr,
    mode="lines",
    name="Rolling Corr",
    line=dict(width=2)
))
fig_corr.add_hline(y=0, line_width=1, line_color="black")

fig_corr.update_layout(
    title=f"Rolling Correlation ({window}D): {ticker1} vs {ticker2} (USD/Foreign)",
    xaxis_title="Date",
    yaxis_title="Correlation",
    template="plotly_white",
    height=450
)

st.plotly_chart(fig_corr, use_container_width=True)

# --------------------------
# 5-Year FX Basket Correlation Matrix (USD/Foreign)
# --------------------------
st.subheader("🌍 5-Year FX Return Correlation Matrix (USD/Foreign Convention)")

direct_pairs = ["SGD=X", "JPY=X", "HKD=X", "CAD=X", "CHF=X", "MXN=X", "PHP=X"]
inverse_pairs = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X"]
all_pairs = direct_pairs + inverse_pairs

# Human-readable labels (all as USD/XXX)
ticker_to_label = {
    "SGD=X": "USD/SGD",
    "JPY=X": "USD/JPY",
    "HKD=X": "USD/HKD",
    "CAD=X": "USD/CAD",
    "CHF=X": "USD/CHF",
    "MXN=X": "USD/MXN",
    "PHP=X": "USD/PHP",
    "EURUSD=X": "USD/EUR",
    "GBPUSD=X": "USD/GBP",
    "AUDUSD=X": "USD/AUD",
}

try:
    df_raw = yf.download(all_pairs, period="5y", auto_adjust=True)["Close"]
except Exception as e:
    st.error(f"Error downloading basket data: {e}")
    st.stop()

if df_raw.empty:
    st.error("No data returned for FX basket.")
    st.stop()

if isinstance(df_raw, pd.Series):
    df_raw = df_raw.to_frame()

df_raw = df_raw.reindex(columns=all_pairs)
missing = df_raw.columns[df_raw.isnull().all()].tolist()
if missing:
    st.warning(f"Missing data for: {missing}")

# Apply same standardization: invert inverse_pairs
for pair in inverse_pairs:
    if pair in df_raw.columns:
        df_raw[pair] = 1 / df_raw[pair]

df_returns = df_raw.pct_change().dropna()

if df_returns.empty:
    st.error("Insufficient data to compute correlations.")
    st.stop()

corr_matrix = df_returns.corr()
display_labels = [ticker_to_label.get(col, col) for col in corr_matrix.columns]
text_annot = corr_matrix.applymap(lambda x: f"{x:.2f}").values

fig_heatmap = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=display_labels,
    y=display_labels,
    colorscale='RdBu',
    zmid=0,
    colorbar=dict(title="Correlation", titleside="right"),
    hoverongaps=False,
    hovertemplate=(
        "<b>FX 1</b>: %{y}<br>"
        "<b>FX 2</b>: %{x}<br>"
        "<b>Correlation</b>: %{z:.3f}<extra></extra>"
    ),
    text=text_annot,
    texttemplate="%{text}",
    textfont={"size": 10}
))

fig_heatmap.update_layout(
    title="5-Year FX Return Correlation Matrix (USD/Foreign)",
    xaxis_title="Currency Pair",
    yaxis_title="Currency Pair",
    template="plotly_white",
    height=700,
    xaxis=dict(tickangle=45),
    yaxis=dict(autorange='reversed')
)

st.plotly_chart(fig_heatmap, use_container_width=True)
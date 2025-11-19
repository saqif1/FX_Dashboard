import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="FX Dashboard", layout="wide")

st.title("FX Strategy Dashboard")
st.write("Experimental Dashboard")

# --------------------------
# Helper: Standardize to USD/Foreign
# --------------------------
def standardize_to_usd_base(series, ticker):
    foreign_per_usd_tickers = {
        "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X",
        "CADUSD=X", "CHFUSD=X"
    }
    if ticker in foreign_per_usd_tickers:
        return 1 / series
    else:
        return series

# --------------------------
# Create Tabs
# --------------------------
tab1, tab2, tab3 = st.tabs(["🔍 FX Correlations", "📉 WC Chart", "🚀 Market Colour"])

# ==============================================================================
# TAB 1: FX CORRELATIONS
# ==============================================================================
with tab1:
    st.subheader("Rolling FX Returns Correlation (USD/Foreign Convention)")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        ticker1 = st.text_input("FX Ticker 1", value="EURUSD=X", key="t1_corr")

    with col2:
        ticker2 = st.text_input("FX Ticker 2", value="GBPUSD=X", key="t2_corr")

    with col3:
        window = st.number_input("Rolling Window (days)", value=30, min_value=5, max_value=365, key="win_corr")

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

    s1 = standardize_to_usd_base(s1_raw, ticker1)
    s2 = standardize_to_usd_base(s2_raw, ticker2)

    df = pd.concat([s1, s2], axis=1)
    df.columns = [ticker1, ticker2]
    df = df.dropna()

    if df.empty:
        st.error("No overlapping data after standardization.")
        st.stop()

    # Rolling Correlation
    returns1 = df[ticker1].pct_change()
    returns2 = df[ticker2].pct_change()
    rolling_corr = returns1.rolling(window).corr(returns2)

    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(
        x=rolling_corr.index,
        y=rolling_corr,
        mode="lines",
        name="Rolling Correlation",
        line=dict(width=2)
    ))
    fig_corr.add_hline(y=0, line_width=1, line_color="black")

    fig_corr.update_layout(
        title=f"Rolling {window}-Day Correlation: {ticker1} vs {ticker2} (USD/Foreign Convention)",
        xaxis_title="Date",
        yaxis_title="Correlation",
        template="plotly_white",
        height=450
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    # Basket Correlation Matrix
    st.subheader("🌍 5-Year FX Return Correlation Matrix (USD/Foreign Convention)")

    direct_pairs = ["SGD=X", "JPY=X", "HKD=X", "CAD=X", "CHF=X", "MXN=X", "PHP=X"]
    inverse_pairs = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X"]
    all_pairs = direct_pairs + inverse_pairs

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
        st.error(f"Error downloading basket: {e}")
        st.stop()

    if df_raw.empty:
        st.error("No data for basket.")
        st.stop()

    if isinstance(df_raw, pd.Series):
        df_raw = df_raw.to_frame()

    df_raw = df_raw.reindex(columns=all_pairs)
    for pair in inverse_pairs:
        if pair in df_raw.columns:
            df_raw[pair] = 1 / df_raw[pair]

    df_returns = df_raw.pct_change().dropna()
    if df_returns.empty:
        st.error("Insufficient data for correlations.")
        st.stop()

    corr_matrix = df_returns.corr()
    display_labels = [ticker_to_label.get(col, col) for col in corr_matrix.columns]
    text_annot = corr_matrix.applymap(lambda x: f"{x:.2f}").values

    fig_heatmap = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=display_labels,
        y=display_labels,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="Correlation"),
        text=text_annot,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Corr: %{z:.3f}<extra></extra>"
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

    

# ==============================================================================
# TAB 2: FAN CHART FORECAST
# ==============================================================================
with tab2:
    st.subheader("🎯 Worst-Case Scenario Chart (Geometric Brownian Motion)")

    # Allow user to choose ticker (default EUR/USD)
    fx_choice = st.selectbox(
        "Select FX Pair (as quoted on Yahoo Finance)",
        options=["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDJPY=X", "USDCAD=X"],
        index=0
    )

    @st.cache_data
    def get_spot_and_vol(ticker, days=252):
        try:
            data = yf.download(ticker, period=f"{days}d", auto_adjust=True, progress=False)["Close"].dropna()
            if len(data) < 30:
                raise ValueError("Insufficient data")
            S0 = data.iloc[-1]
            log_ret = np.log(data / data.shift(1)).dropna()
            sigma = log_ret.std() * np.sqrt(252)
            return float(S0), float(sigma), True
        except Exception as e:
            return 1.08, 0.10, False

    S0, sigma, success = get_spot_and_vol(fx_choice)

    if not success:
        st.warning(f"Using default values (S₀={S0:.3f}, σ={sigma:.0%}) due to data issues.")

    # GBM Parameters
    mu = 0.0  # drift = 0 for FX risk-neutral or uncertainty bands
    horizons_months = np.array([0, 3, 6, 9, 12])  # <-- INCLUDE t=0
    T = horizons_months / 12.0
    percentiles = [0.5, 0.75, 0.85, 0.95]
    z_scores = norm.ppf(percentiles)

    # Compute fan: include t=0 explicitly
    fan_data = {}
    for t, month in zip(T, horizons_months):
        if t == 0:
            # At t=0, all percentiles = S0 (no uncertainty)
            spots = np.full(len(percentiles), S0)
        else:
            base = (mu - 0.5 * sigma**2) * t
            vol_term = sigma * np.sqrt(t)
            spots = S0 * np.exp(base + vol_term * z_scores)
        fan_data[month] = spots

    df_fan = pd.DataFrame(fan_data, index=[f"P{int(p*100)}" for p in percentiles]).T

    # Plot with Plotly
    fig_fan = go.Figure()

    # Keep the median
    fig_fan.add_trace(go.Scatter(
        x=df_fan.index,
        y=df_fan['P50'],
        mode='lines',
        name='Median (50%)',
        line=dict(color='red', width=2)
    ))

    # Add upper percentiles as simple lines (no fill)
    fig_fan.add_trace(go.Scatter(
        x=df_fan.index,
        y=df_fan['P75'],
        mode='lines',
        name='75th Percentile',
        line=dict(color='gold', dash='dot')
    ))

    fig_fan.add_trace(go.Scatter(
        x=df_fan.index,
        y=df_fan['P85'],
        mode='lines',
        name='85th Percentile',
        line=dict(color='indianred', dash='dot')
    ))

    fig_fan.add_trace(go.Scatter(
        x=df_fan.index,
        y=df_fan['P95'],
        mode='lines',
        name='95th Percentile',
        line=dict(color='darkblue', dash='dot')
    ))

    # Optional: upper percentile lines
    # for p in ['P75', 'P85', 'P95']:
    #     fig_fan.add_trace(go.Scatter(
    #         x=df_fan.index, y=df_fan[p],
    #         mode='lines',
    #         line=dict(color='blue', dash='dot', width=1),
    #         showlegend=False,
    #         hoverinfo='skip'
    #     ))

    # Define numeric positions (in months) — now includes 0
    horizons_months_plot = [0, 3, 6, 9, 12]
    x_labels = ["Spot", "3M later", "6M later", "9M later", "1Y later"]

    fig_fan.update_layout(
        title=f"FX Fan Chart – GBM Forecast<br>Pair: {fx_choice} | Spot: {S0:.5f} | Vol: {sigma:.2%}",
        xaxis_title="Horizon",
        yaxis_title=fx_choice,
        template="plotly_white",
        height=500,
        xaxis=dict(
            tickmode='array',
            tickvals=horizons_months_plot,
            ticktext=x_labels,
            range=[-0.5, 12.5]  # ensures full visibility of Spot at 0
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_fan, use_container_width=True)

    st.info("""
    **Methodology**:  
    - **Spot rate (`S₀`)**: Today’s observed market price (most recent closing rate).  
    - **Volatility (`σ`)**: Annualised historical volatility computed from daily log returns over the past 1 year (252 trading days).  
    - **Model**: Geometric Brownian Motion (GBM) with **zero drift** — meaning no assumed directional trend, only uncertainty from volatility.  
    - **Percentiles** (e.g. 50%, 75%, 85%, 95%) show the statistical distribution of possible future rate paths under this model.  
    → This is **not a price prediction**, but a visualisation of **potential uncertainty** around the current spot rate.
    """)
# ==============================================================================
# TAB 2: MARKET COLOUR
# ==============================================================================
with tab3:
    st.subheader("Market Colour")

    st.error("To test LMM Model..")


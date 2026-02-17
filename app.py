import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations

st.set_page_config(page_title="GrokMedallion 3.0", layout="wide")
st.title("🦾 GrokMedallion Fund 3.0")
st.markdown("**Medallion Simulator v3** – Stat Arb + Regime Detection. Short-term edges.")

# ────────────────────────────────────────────────
# Sidebar Controls
# ────────────────────────────────────────────────
st.sidebar.header("Controls")
universe = st.sidebar.multiselect(
    "Select Assets",
    ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ', 'TSLA', 'IWM'],
    default=['AAPL', 'MSFT', 'NVDA', 'SPY', 'QQQ']
)
period = st.sidebar.selectbox("Backtest Period", ['3mo', '6mo', '1y', '2y'], index=2)

# ────────────────────────────────────────────────
# Data Fetch – FIXED for yfinance MultiIndex / column changes
# ────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_data(tickers, period):
    if not tickers:
        st.error("No assets selected.")
        return pd.DataFrame()

    try:
        # Download with auto_adjust to get adjusted prices in 'Close'
        df = yf.download(
            tickers,
            period=period,
            interval='1d',
            progress=False,
            auto_adjust=True,
            repair=True
        )

        if df.empty:
            st.error("yfinance returned empty data. Try a shorter period or fewer symbols.")
            return pd.DataFrame()

        # ── Handle column structure ───────────────────────────────
        if isinstance(df.columns, pd.MultiIndex):
            # Multi-ticker: prefer 'Close' (adjusted when auto_adjust=True)
            if 'Close' in df.columns.levels[0]:
                prices = df.xs('Close', level='Price', axis=1, drop_level=True)
            # Fallback if only 'Adj Close' exists (rare with auto_adjust)
            elif 'Adj Close' in df.columns.levels[0]:
                prices = df.xs('Adj Close', level='Price', axis=1, drop_level=True)
            else:
                st.error("No usable price column ('Close' or 'Adj Close') found in multi-ticker data.")
                return pd.DataFrame()
        else:
            # Single ticker or flat structure
            if 'Close' in df.columns:
                prices = df['Close']
            elif 'Adj Close' in df.columns:
                prices = df['Adj Close']
            else:
                st.error("No price data available (missing 'Close' and 'Adj Close').")
                return pd.DataFrame()

            # Convert Series → DataFrame with correct column name
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers if isinstance(tickers, str) else tickers[0])

        # Clean data
        prices = prices.dropna(how='all').dropna(how='any', axis=0)

        if prices.empty:
            st.warning("After cleaning, no valid price data remains.")
            return pd.DataFrame()

        return prices

    except Exception as e:
        st.error(f"yfinance download failed: {str(e)}\n\nSuggestions:\n• Select 2–5 symbols max\n• Try shorter period (e.g. '3mo')\n• Check internet connection")
        return pd.DataFrame()

# Fetch data
data = fetch_data(universe, period)

if data.empty:
    st.stop()

# ────────────────────────────────────────────────
# Signal Generation
# ────────────────────────────────────────────────
def generate_signals(data):
    try:
        signals = pd.DataFrame(index=data.index)
        pairs = list(combinations(data.columns, 2))[:10]  # Limit pairs to prevent slowdown

        for a, b in pairs:
            hedge = data[a].mean() / data[b].mean()
            spread = data[a] - hedge * data[b]
            roll_mean = spread.rolling(window=20, min_periods=1).mean()
            roll_std = spread.rolling(window=20, min_periods=1).std()
            z = (spread - roll_mean) / roll_std

            rets = data[a].pct_change().fillna(0)
            vol = rets.rolling(10, min_periods=1).std().fillna(0)
            regime = np.where(
                (rets > rets.mean()) & (vol < vol.mean()), 1,
                np.where((rets < rets.mean()) & (vol > vol.mean()), -1, 0)
            )

            sig = np.where((z > 1.5) & (regime == 1), 1,
                           np.where((z < -1.5) & (regime == 1), -1, 0))
            signals[f'{a}-{b}'] = sig

        final = signals.mean(axis=1)
        final = np.where(final > 0.2, 1, np.where(final < -0.2, -1, 0))
        return pd.DataFrame({'signal': final, 'strength': np.abs(final)}, index=data.index)

    except Exception as e:
        st.error(f"Error generating signals: {str(e)}")
        return pd.DataFrame()

signals = generate_signals(data)

# ────────────────────────────────────────────────
# Backtest Logic
# ────────────────────────────────────────────────
def backtest(data, signals):
    positions = signals['signal'].shift(1).fillna(0)
    rets = data.pct_change().mean(axis=1).fillna(0)
    strat_rets = positions * rets * 5  # 5x leverage
    costs = positions.diff().abs() * 0.0005  # 5 bps transaction cost
    strat_rets -= costs
    equity = (1 + strat_rets).cumprod()
    sharpe = strat_rets.mean() / strat_rets.std() * np.sqrt(252) if strat_rets.std() != 0 else 0
    dd = (equity / equity.cummax() - 1).min()
    return equity, sharpe, dd

equity, sharpe, dd = backtest(data, signals)

# ────────────────────────────────────────────────
# UI – Tabs
# ────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Backtest", "Live Signals"])

with tab1:
    st.subheader("Backtest Results")
    col1, col2 = st.columns(2)
    col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col2.metric("Max Drawdown", f"{dd*100:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y

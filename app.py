import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
import time

st.set_page_config(page_title="GrokMedallion 3.0", layout="wide")
st.title("🦾 GrokMedallion Fund 3.0")
st.markdown("**Medallion Simulator v3** – Stat Arb + Regime Detection. Short-term edges.")

# Sidebar
st.sidebar.header("Controls")
universe = st.sidebar.multiselect(
    "Select Assets (need at least 2 for pairs)",
    ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ', 'TSLA', 'IWM'],
    default=['AAPL', 'SPY']
)
period = st.sidebar.selectbox("Backtest Period", ['3mo', '6mo', '1y', '2y'], index=2)  # Default 1y

# Data fetch with retry
@st.cache_data(ttl=300)
def fetch_data(tickers, period):
    if len(tickers) < 1:
        st.error("Select at least 1 asset.")
        return pd.DataFrame()

    try:
        df = yf.download(tickers, period=period, interval='1d', progress=False, auto_adjust=True, repair=True, threads=False)

        if df.empty:
            st.info("First fetch empty – retrying...")
            time.sleep(3)
            df = yf.download(tickers, period=period, interval='1d', progress=False, auto_adjust=True, repair=True, threads=False)

        if df.empty:
            st.warning("No data after retry. Try '2y' or different symbols.")
            return pd.DataFrame()

        # Column handling
        if isinstance(df.columns, pd.MultiIndex):
            prices = df.xs('Close', level='Price', axis=1, drop_level=True) if 'Close' in df.columns.levels[0] else df.xs('Adj Close', level='Price', axis=1, drop_level=True)
        else:
            prices = df['Close'] if 'Close' in df.columns else df['Adj Close']
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0] if len(tickers) == 1 else 'Price')

        prices = prices.dropna(how='all').dropna(axis=0, how='any')
        return prices

    except Exception as e:
        st.error(f"Fetch error: {str(e)}")
        return pd.DataFrame()

data = fetch_data(universe, period)

if data.empty:
    st.stop()

# Debug data
st.sidebar.write("Data rows:", len(data))
st.sidebar.write("Columns:", list(data.columns))

# Signals with debug
def generate_signals(data):
    if len(data.columns) < 2:
        st.warning("Need at least 2 assets for pairs trading signals.")
        return pd.DataFrame()

    try:
        signals = pd.DataFrame(index=data.index)
        pairs = list(combinations(data.columns, 2))[:10]

        st.sidebar.info(f"Testing {len(pairs)} pairs")

        for a, b in pairs:
            hedge = data[a].mean() / data[b].mean()
            spread = data[a] - hedge * data[b]
            roll_mean = spread.rolling(20, min_periods=5).mean()
            roll_std = spread.rolling(20, min_periods=5).std()
            z = (spread - roll_mean) / roll_std

            rets = data[a].pct_change().fillna(0)
            vol = rets.rolling(10, min_periods=5).std().fillna(0)
            regime = np.where(
                (rets > rets.mean()) & (vol < vol.mean()), 1,
                np.where((rets < rets.mean()) & (vol > vol.mean()), -1, 0)
            )

            sig = np.where((z > 1.5) & (regime == 1), 1,
                           np.where((z < -1.5) & (regime == 1), -1, 0))
            signals[f'{a}-{b}'] = sig

        final = signals.mean(axis=1)
        st.sidebar.write("Raw final signal mean:", final.mean())

        final = np.where(final > 0.2, 1, np.where(final < -0.2, -1, 0))
        return pd.DataFrame({'signal': final, 'strength': np.abs(final)}, index=data.index)

    except Exception as e:
        st.error(f"Signals error: {str(e)}")
        return pd.DataFrame()

signals = generate_signals(data)

# Backtest
def backtest(data, signals):
    if signals.empty:
        return pd.Series(), 0, 0

    positions = signals['signal'].shift(1).fillna(0)
    rets = data.pct_change().mean(axis=1).fillna(0)
    strat_rets = positions * rets * 5
    costs = positions.diff().abs() * 0.0005
    strat_rets -= costs
    equity = (1 + strat_rets).cumprod()
    sharpe = strat_rets.mean() / strat_rets.std() * np.sqrt(252) if strat_rets.std() != 0 else 0
    dd = (equity / equity.cummax() - 1).min()
    return equity, sharpe, dd

equity, sharpe, dd = backtest(data, signals)

# UI
tab1, tab2 = st.tabs(["Backtest", "Live Signals"])

with tab1:
    st.subheader("Backtest Results")
    col1, col2 = st.columns(2)
    col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col2.metric("Max Drawdown", f"{dd*100:.1f}%")

    if not equity.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity.index, y=equity, mode='lines', name='Equity Curve'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No equity curve – no signals generated.")

with tab2:
    st.subheader("Current Signals")
    if signals.empty:
        st.warning("No signals generated. Check debug info in sidebar.")
    else:
        st.dataframe(signals.tail(10), use_container_width=True)

    active = len(signals[signals['signal'] != 0]) if not signals.empty else 0
    if active > 0:
        st.success(f"{active} active signals right now!")
    else:
        st.info("No strong signals currently.")

st.caption("GrokMedallion 3.0 – Refresh page. Sidebar debug info helps diagnose empty signals.")

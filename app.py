import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations

st.set_page_config(page_title="GrokMedallion 3.0", layout="wide")
st.title("🦾 GrokMedallion Fund 3.0")
st.markdown("**Medallion Simulator v3** – Stat Arb + Regime Detection. Short-term edges.")

# Sidebar
st.sidebar.header("Controls")
universe = st.sidebar.multiselect(
    "Select Assets",
    ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ', 'TSLA', 'IWM'],
    default=['AAPL', 'MSFT', 'NVDA', 'SPY', 'QQQ']
)
period = st.sidebar.selectbox("Backtest Period", ['3mo', '6mo', '1y', '2y'], index=2)

# Data fetch – robust handling for yfinance column structure
@st.cache_data(ttl=300)
def fetch_data(tickers, period):
    if not tickers:
        st.error("No assets selected.")
        return pd.DataFrame()

    try:
        df = yf.download(
            tickers,
            period=period,
            interval='1d',
            progress=False,
            auto_adjust=True,
            repair=True
        )

        if df.empty:
            st.error("yfinance returned empty data.")
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.levels[0]:
                prices = df.xs('Close', level='Price', axis=1, drop_level=True)
            elif 'Adj Close' in df.columns.levels[0]:
                prices = df.xs('Adj Close', level='Price', axis=1, drop_level=True)
            else:
                st.error("No 'Close' or 'Adj Close' in multi-ticker data.")
                return pd.DataFrame()
        else:
            if 'Close' in df.columns:
                prices = df['Close']
            elif 'Adj Close' in df.columns:
                prices = df['Adj Close']
            else:
                st.error("No price column found.")
                return pd.DataFrame()

            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers if isinstance(tickers, str) else tickers[0])

        prices = prices.dropna(how='all').dropna(how='any', axis=0)

        if prices.empty:
            st.warning("No valid price data after cleaning.")
            return pd.DataFrame()

        return prices

    except Exception as e:
        st.error(f"yfinance error: {str(e)}\nTry 2–4 symbols or shorter period.")
        return pd.DataFrame()

data = fetch_data(universe, period)

if data.empty:
    st.stop()

# Generate signals
def generate_signals(data):
    try:
        signals = pd.DataFrame(index=data.index)
        pairs = list(combinations(data.columns, 2))[:10]

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
        st.error(f"Signal error: {str(e)}")
        return pd.DataFrame()

signals = generate_signals(data)

# Backtest
def backtest(data, signals):
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

# UI Tabs
tab1, tab2 = st.tabs(["Backtest", "Live Signals"])

with tab1:
    st.subheader("Backtest Results")
    col1, col2 = st.columns(2)
    col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col2.metric("Max Drawdown", f"{dd*100:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity, mode='lines', name='Equity Curve'))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Current Signals")
    st.dataframe(signals.tail(10), use_container_width=True)
    active = len(signals[signals['signal'] != 0])
    if active > 0:
        st.success(f"{active} active signals right now!")
    else:
        st.info("No strong signals currently.")

st.caption("GrokMedallion 3.0 – Refresh for latest data.")

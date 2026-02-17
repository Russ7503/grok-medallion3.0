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

# Data fetch – FIXED for yfinance MultiIndex columns
@st.cache_data(ttl=300)
def fetch_data(tickers, period):
    try:
        df = yf.download(tickers, period=period, interval='1d', progress=False)
        
        if df.empty:
            st.error("No data returned from yfinance.")
            return pd.DataFrame()
        
        # Handle MultiIndex (new yfinance behavior for multiple tickers)
        if isinstance(df.columns, pd.MultiIndex):
            # Extract only Adjusted Close prices
            adj_close = df.xs('Adj Close', level='Price', axis=1, drop_level=True)
        else:
            # Single ticker fallback
            adj_close = df['Adj Close'].to_frame(name=tickers[0])
        
        return adj_close.dropna(how='all')
    
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}. Try fewer assets or a different period.")
        return pd.DataFrame()

data = fetch_data(universe, period)

if data.empty:
    st.stop()

# Signals function
def generate_signals(data):
    try:
        signals = pd.DataFrame(index=data.index)
        pairs = list(combinations(data.columns, 2))[:10]  # Limit to avoid slowdown
        
        for a, b in pairs:
            hedge = data[a].mean() / data[b].mean()
            spread = data[a] - hedge * data[b]
            roll_mean = spread.rolling(window=20, min_periods=1).mean()
            roll_std = spread.rolling(window=20, min_periods=1).std()
            z = (spread - roll_mean) / roll_std
            
            rets = data[a].pct_change().fillna(0)
            vol = rets.rolling(10, min_periods=1).std().fillna(0)
            regime = np.where(
                (rets > rets.mean()) & (vol < vol.mean()), 1,   # Bull
                np.where((rets < rets.mean()) & (vol > vol.mean()), -1, 0)  # Bear/Chop
            )
            
            sig = np.where((z > 1.5) & (regime == 1), 1,
                           np.where((z < -1.5) & (regime == 1), -1, 0))
            signals[f'{a}-{b}'] = sig
        
        final = signals.mean(axis=1)
        final = np.where(final > 0.2, 1, np.where(final < -0.2, -1, 0))
        return pd.DataFrame({'signal': final, 'strength': np.abs(final)}, index=data.index)
    except Exception as e:
        st.error(f"Signal generation error: {str(e)}")
        return pd.DataFrame()

signals = generate_signals(data)

# Backtest
def backtest(data, signals):
    positions = signals['signal'].shift(1).fillna(0)
    rets = data.pct_change().mean(axis=1).fillna(0)
    strat_rets = positions * rets * 5  # 5x leverage
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

st.caption("GrokMedallion 3.0 – Built step-by-step. Refresh page for latest data.")

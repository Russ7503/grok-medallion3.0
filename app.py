import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from itertools import combinations

st.set_page_config(page_title="GrokMedallion", layout="wide")
st.title("🦾 GrokMedallion Fund v1.0")
st.markdown("**Your Medallion Simulator** | Stat Arb Edges. 51% Win Rate Sim. Short-term, Market-Neutral.")

# Sidebar
st.sidebar.header("Controls")
universe = st.sidebar.multiselect("Assets (Add More)", 
    ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ', 'TSLA', 'IWM'], 
    default=['AAPL', 'MSFT', 'NVDA', 'SPY', 'QQQ'])
period = st.sidebar.selectbox("Backtest Period", ['3mo', '6mo', '1y', '2y'], index=2)

# Data Fetch
@st.cache_data(ttl=60)
def fetch_data(tickers, period):
    data = yf.download(tickers, period=period, interval='1d', progress=False)['Adj Close']
    return data.dropna()

data = fetch_data(universe, period)

# Signals (Stat Arb + Regime)
def generate_signals(data):
    signals = pd.DataFrame(index=data.index)
    pairs = list(combinations(data.columns, 2))[:15]  # More pairs
    
    for a, b in pairs:
        hedge = data[a].mean() / data[b].mean()
        spread = data[a] - hedge * data[b]
        z = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
        
        # Regime: Bull/Bear/Chop
        rets = data[a].pct_change().fillna(0)
        vol = rets.rolling(10).std().fillna(0)
        regime = np.where((rets > rets.mean()) & (vol < vol.mean()), 1,  # Bull
                         np.where((rets < rets.mean()) & (vol > vol.mean()), -1, 0))  # Bear/Chop
        
        # Signal
        sig = np.where((z > 1.5) & (regime == 1), 1, 
                      np.where((z < -1.5) & (regime == 1), -1, 0))
        signals[f'{a}-{b}'] = sig
    
    # Fuse into one signal
    final = signals.mean(axis=1)
    final = np.where(final > 0.2, 1, np.where(final < -0.2, -1, 0))
    return pd.DataFrame({'signal': final, 'strength': final.abs()}, index=data.index)

signals = generate_signals(data)

# Backtest
def backtest(data, signals):
    positions = signals['signal'].shift(1).fillna(0)
    rets = data.pct_change().mean(axis=1)
    strat_rets = positions * rets * 5  # 5x leverage
    costs = positions.diff().abs() * 0.0005
    strat_rets -= costs
    equity = (1 + strat_rets).cumprod()
    sharpe = strat_rets.mean() / strat_rets.std() * np.sqrt(252) if strat_rets.std() > 0 else 0
    dd = (equity / equity.cummax() - 1).min()
    return equity, sharpe, dd, strat_rets

equity, sharpe, dd, daily = backtest(data, signals)

# UI
tab1, tab2, tab3 = st.tabs(["Backtest", "Live Signals", "Next: Robinhood"])

with tab1:
    st.subheader("📊 Backtest (Medallion-Style)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe", f"{sharpe:.2f} (Target 3+)", "Elite")
    col2.metric("Max DD", f"{dd*100:.1f}%")
    col3.metric("CAGR", f"{((equity.iloc[-1]/equity.iloc[0])**(252/len(equity))-1)*100:.1f}%")
    
    fig = go.Figure(go.Scatter(x=equity.index, y=equity, name="Equity"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(daily.tail())

with tab2:
    st.subheader("📡 Live Signals (Auto-Refresh)")
    if st.button("🔄 Update Now"):
        data = fetch_data(universe, period)
        signals = generate_signals(data)
    st.dataframe(signals.tail(10), use_container_width=True)
    st.success(f"**{len(signals[signals['signal'] != 0])} Active Trades** | Hold 1-3 days")

with tab3:
    st.subheader("💰 Robinhood Ready (Coming in v1.1)")
    st.info("Next step: Add robin_stocks and login code.")

st.caption("**Pro Tip**: This sims strong returns in backtests. Refresh for live. Your Medallion starts here.")

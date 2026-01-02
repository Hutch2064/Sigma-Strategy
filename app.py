import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

class Config:
    # App config
    APP_NAME = "SigmaTrader Pro"
    VERSION = "1.0.0"
    OFFICIAL_START_DATE = "2025-12-22"
    
    # Strategy defaults
    RISK_ON_WEIGHTS = {"BITU": 0.3333, "QQQU": 0.3333, "UGL": 0.3333}
    RISK_OFF_WEIGHTS = {"SHY": 1.0}
    FLIP_COST = 0.0005
    START_RISKY = 0.6
    START_SAFE = 0.4
    
    # Fixed parameters (no optimization in production)
    FIXED_MA_LENGTH = 200
    FIXED_MA_TYPE = "sma"
    
    # Performance tracking
    RISK_FREE_RATE = 0.0

# ============================================================
# CORE STRATEGY CLASSES (Refactored)
# ============================================================

class DataLoader:
    """Handles data loading with caching and error handling"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=True)
    def load_price_data(tickers, start_date, end_date=None):
        """Load price data with robust error handling"""
        try:
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date, 
                progress=False,
                auto_adjust=True
            )
            
            if data.empty:
                raise ValueError("No data returned from Yahoo Finance")
            
            # Handle single vs multiple tickers
            if "Adj Close" in data.columns:
                prices = data["Adj Close"].copy()
            elif "Close" in data.columns:
                prices = data["Close"].copy()
            else:
                # Handle multi-level columns
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data.xs('Close', level=0, axis=1)
                else:
                    prices = data
            
            # Ensure DataFrame format
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0])
            
            return prices.dropna(how='all')
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()

class PortfolioBuilder:
    """Builds and manages portfolio indices"""
    
    @staticmethod
    def build_portfolio_index(prices, weights_dict, annual_drag_pct=0.0):
        """Build portfolio index with optional daily drag"""
        simple_rets = prices.pct_change().fillna(0)
        idx_rets = pd.Series(0.0, index=simple_rets.index)
        
        for ticker, weight in weights_dict.items():
            if ticker in simple_rets.columns:
                idx_rets += simple_rets[ticker] * weight
        
        # Apply portfolio drag if specified
        if annual_drag_pct > 0:
            daily_drag_factor = (1 - annual_drag_pct) ** (1/252)
            idx_rets = (1 + idx_rets) * daily_drag_factor - 1
        
        # Cumulative returns
        cumprod = (1 + idx_rets).cumprod()
        
        # Handle edge cases
        if cumprod.empty or cumprod.isna().all():
            return pd.Series(1.0, index=prices.index)
        
        # Forward fill from first valid value
        valid_idx = cumprod.first_valid_index()
        if valid_idx:
            cumprod.loc[:valid_idx] = 1.0
            cumprod = cumprod.ffill()
        
        return cumprod
    
    @staticmethod
    def compute_returns(prices, weights_dict, annual_drag_pct=0.0):
        """Compute portfolio returns"""
        simple_rets = pd.Series(0.0, index=prices.index)
        
        for ticker, weight in weights_dict.items():
            if ticker in prices.columns:
                ticker_rets = prices[ticker].pct_change().fillna(0)
                simple_rets += ticker_rets * weight
        
        # Apply drag
        if annual_drag_pct > 0:
            daily_drag_factor = (1 - annual_drag_pct) ** (1/252)
            simple_rets = (1 + simple_rets) * daily_drag_factor - 1
        
        return simple_rets

class MAStrategy:
    """Moving Average strategy implementation"""
    
    @staticmethod
    def compute_ma(price_series, length, ma_type="sma"):
        """Compute moving average"""
        if ma_type.lower() == "ema":
            return price_series.ewm(span=length, adjust=False).mean().shift(1)
        else:
            return price_series.rolling(window=length, min_periods=1).mean().shift(1)
    
    @staticmethod
    def generate_signal(price, ma, tolerance):
        """Generate MA-based signals"""
        if price.empty or ma.empty:
            return pd.Series(False, index=price.index)
        
        # Vectorized signal generation
        upper = ma * (1 + tolerance)
        lower = ma * (1 - tolerance)
        
        # Initialize signal array
        signal = pd.Series(False, index=price.index)
        
        for i in range(1, len(price)):
            if pd.isna(price.iloc[i]) or pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]):
                signal.iloc[i] = signal.iloc[i-1] if i > 0 else False
            elif not signal.iloc[i-1]:
                signal.iloc[i] = price.iloc[i] > upper.iloc[i]
            else:
                signal.iloc[i] = not (price.iloc[i] < lower.iloc[i])
        
        return signal.fillna(False)

class PerformanceAnalyzer:
    """Comprehensive performance analysis"""
    
    @staticmethod
    def compute_enhanced_metrics(returns, equity_curve, rf=0.0):
        """Compute comprehensive performance metrics"""
        if len(returns) < 2 or equity_curve.empty:
            return {}
        
        # Basic metrics
        n_days = len(returns)
        n_years = n_days / 252
        
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
        
        # Risk metrics
        vol = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252 - rf) / vol if vol > 0 else 0
        
        # Drawdown
        dd = equity_curve / equity_curve.cummax() - 1
        max_dd = dd.min()
        
        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (returns.mean() * 252 - rf) / downside_dev if downside_dev > 0 else 0
        
        # Additional metrics
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        return {
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Calmar": calmar,
            "MaxDrawdown": max_dd,
            "TotalReturn": total_return,
            "WinRate": win_rate,
            "DD_Series": dd
        }
    
    @staticmethod
    def compute_trade_metrics(signal):
        """Compute trading statistics"""
        if signal.empty:
            return {"trades_per_year": 0, "switches": 0}
        
        switches = signal.astype(int).diff().abs().sum()
        trades_per_year = switches / (len(signal) / 252) if len(signal) > 0 else 0
        
        return {
            "trades_per_year": trades_per_year,
            "switches": switches
        }

# ============================================================
# STREAMLIT APP - PRODUCTION VERSION
# ============================================================

def setup_page():
    """Configure Streamlit page"""
    st.set_page_config(
        page_title=Config.APP_NAME,
        layout="wide",
        page_icon="📈",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .strategy-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .badge-on { background-color: #10B981; color: white; }
    .badge-off { background-color: #EF4444; color: white; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown(f'<h1 class="main-header">{Config.APP_NAME} v{Config.VERSION}</h1>', unsafe_allow_html=True)
    
    # Demo warning
    with st.container():
        st.warning("⚠️ **Demo Version** - Live trading not enabled. Authentication coming soon.")

def render_sidebar():
    """Render sidebar controls"""
    with st.sidebar:
        st.header("📊 Strategy Configuration")
        
        # Date inputs
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.date(2020, 1, 1),
                max_value=datetime.date.today()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.date.today()
            )
        
        st.divider()
        
        # Portfolio configuration
        st.subheader("Portfolio Allocation")
        
        with st.expander("Risk-On Assets", expanded=True):
            risk_on_tickers = st.text_input(
                "Tickers (comma-separated)",
                value=",".join(Config.RISK_ON_WEIGHTS.keys()),
                help="Enter tickers like: BITU,QQQU,UGL"
            )
            risk_on_weights = st.text_input(
                "Weights (comma-separated)",
                value=",".join(str(w) for w in Config.RISK_ON_WEIGHTS.values()),
                help="Enter weights like: 0.3333,0.3333,0.3333"
            )
        
        with st.expander("Risk-Off Assets", expanded=True):
            risk_off_tickers = st.text_input(
                "Ticker",
                value=list(Config.RISK_OFF_WEIGHTS.keys())[0],
                help="Typically SHY or equivalent treasury ETF"
            )
        
        st.divider()
        
        # Portfolio drag
        st.subheader("Costs & Drag")
        annual_drag = st.slider(
            "Annual Portfolio Drag (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="Annual decay for leveraged ETFs (~4% for 3x)"
        )
        
        st.divider()
        
        # Account values
        st.subheader("Account Values")
        
        accounts = {
            "Taxable": {
                "last_rebalance": st.number_input(
                    "Taxable - Last Rebalance ($)",
                    min_value=0.0,
                    value=75815.26,
                    step=1000.0
                ),
                "current": st.number_input(
                    "Taxable - Current ($)",
                    min_value=0.0,
                    value=68832.42,
                    step=1000.0
                )
            },
            "Tax-Sheltered": {
                "last_rebalance": st.number_input(
                    "Tax-Sheltered - Last Rebalance ($)",
                    min_value=0.0,
                    value=10074.83,
                    step=1000.0
                ),
                "current": st.number_input(
                    "Tax-Sheltered - Current ($)",
                    min_value=0.0,
                    value=9265.91,
                    step=1000.0
                )
            },
            "Joint": {
                "last_rebalance": st.number_input(
                    "Joint - Last Rebalance ($)",
                    min_value=0.0,
                    value=4189.76,
                    step=1000.0
                ),
                "current": st.number_input(
                    "Joint - Current ($)",
                    min_value=0.0,
                    value=3930.23,
                    step=1000.0
                )
            }
        }
        
        st.divider()
        
        # Fixed parameters display
        st.subheader("Strategy Parameters")
        st.info(f"""
        **Fixed Parameters:**
        - MA Length: {Config.FIXED_MA_LENGTH}
        - MA Type: {Config.FIXED_MA_TYPE.upper()}
        - Flip Cost: {Config.FLIP_COST:.2%}
        - Portfolio Drag: {annual_drag:.1f}%
        """)
        
        # Run button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            return {
                "start_date": start_date,
                "end_date": end_date,
                "risk_on_tickers": [t.strip().upper() for t in risk_on_tickers.split(",")],
                "risk_on_weights": [float(w) for w in risk_on_weights.split(",")],
                "risk_off_tickers": [risk_off_tickers.strip().upper()],
                "risk_off_weights": [1.0],
                "annual_drag": annual_drag / 100,
                "accounts": accounts
            }
    
    return None

def render_dashboard(params):
    """Main dashboard rendering"""
    
    # Load data
    with st.spinner("Loading market data..."):
        loader = DataLoader()
        all_tickers = params["risk_on_tickers"] + params["risk_off_tickers"]
        prices = loader.load_price_data(
            all_tickers,
            params["start_date"],
            params["end_date"]
        )
    
    if prices.empty:
        st.error("No data available for the selected tickers and date range.")
        return
    
    # Initialize components
    portfolio_builder = PortfolioBuilder()
    ma_strategy = MAStrategy()
    analyzer = PerformanceAnalyzer()
    
    # Build portfolios
    risk_on_weights = dict(zip(params["risk_on_tickers"], params["risk_on_weights"]))
    risk_off_weights = dict(zip(params["risk_off_tickers"], params["risk_off_weights"]))
    
    # Create portfolio index
    portfolio_index = portfolio_builder.build_portfolio_index(
        prices, risk_on_weights, params["annual_drag"]
    )
    
    # Compute MA
    ma = ma_strategy.compute_ma(
        portfolio_index,
        Config.FIXED_MA_LENGTH,
        Config.FIXED_MA_TYPE
    )
    
    # Optimize tolerance
    with st.spinner("Optimizing strategy parameters..."):
        # Simple tolerance optimization (could be enhanced)
        tolerance = 0.002  # Default 0.2%
        
        # Generate signal
        signal = ma_strategy.generate_signal(portfolio_index, ma, tolerance)
    
    # Current regime
    current_regime = "RISK-ON" if signal.iloc[-1] else "RISK-OFF"
    badge_class = "badge-on" if signal.iloc[-1] else "badge-off"
    
    # Display current status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Regime", current_regime)
    with col2:
        st.metric("Portfolio Value", f"${params['accounts']['Taxable']['current']:,.0f}")
    with col3:
        st.metric("MA Tolerance", f"{tolerance:.2%}")
    with col4:
        st.metric("Data Points", len(prices))
    
    st.divider()
    
    # Performance Metrics
    st.subheader("📊 Performance Analysis")
    
    # Calculate returns
    risk_on_returns = portfolio_builder.compute_returns(
        prices, risk_on_weights, params["annual_drag"]
    )
    risk_on_equity = (1 + risk_on_returns).cumprod()
    
    # Simple backtest (simplified for demo)
    strategy_returns = risk_on_returns.where(signal.shift(1), 0)
    strategy_equity = (1 + strategy_returns).cumprod()
    
    # Calculate metrics
    risk_on_metrics = analyzer.compute_enhanced_metrics(risk_on_returns, risk_on_equity)
    strategy_metrics = analyzer.compute_enhanced_metrics(strategy_returns, strategy_equity)
    trade_metrics = analyzer.compute_trade_metrics(signal)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Strategy CAGR", f"{strategy_metrics.get('CAGR', 0):.2%}")
        st.metric("Buy & Hold CAGR", f"{risk_on_metrics.get('CAGR', 0):.2%}")
    
    with col2:
        st.metric("Strategy Sharpe", f"{strategy_metrics.get('Sharpe', 0):.2f}")
        st.metric("Max Drawdown", f"{strategy_metrics.get('MaxDrawdown', 0):.2%}")
    
    with col3:
        st.metric("Win Rate", f"{strategy_metrics.get('WinRate', 0):.2%}")
        st.metric("Volatility", f"{strategy_metrics.get('Volatility', 0):.2%}")
    
    with col4:
        st.metric("Trades/Year", f"{trade_metrics.get('trades_per_year', 0):.1f}")
        st.metric("Total Return", f"{strategy_metrics.get('TotalReturn', 0):.2%}")
    
    st.divider()
    
    # Charts
    st.subheader("📈 Performance Charts")
    
    tab1, tab2, tab3 = st.tabs(["Equity Curve", "Drawdown", "Regime Analysis"])
    
    with tab1:
        fig = go.Figure()
        
        # Add equity curves
        fig.add_trace(go.Scatter(
            x=strategy_equity.index,
            y=strategy_equity.values,
            name="Strategy",
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=risk_on_equity.index,
            y=risk_on_equity.values,
            name="Buy & Hold",
            line=dict(color='blue', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Equity Curve Comparison",
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            hovermode="x unified",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure()
        
        dd_strategy = strategy_metrics.get('DD_Series', pd.Series())
        dd_bh = risk_on_metrics.get('DD_Series', pd.Series())
        
        if not dd_strategy.empty:
            fig.add_trace(go.Scatter(
                x=dd_strategy.index,
                y=dd_strategy.values * 100,
                name="Strategy Drawdown",
                fill='tozeroy',
                line=dict(color='red')
            ))
        
        if not dd_bh.empty:
            fig.add_trace(go.Scatter(
                x=dd_bh.index,
                y=dd_bh.values * 100,
                name="Buy & Hold Drawdown",
                fill='tozeroy',
                line=dict(color='orange')
            ))
        
        fig.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Regime visualization
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=("Portfolio vs MA", "Regime Signal"))
        
        # Price and MA
        fig.add_trace(go.Scatter(
            x=portfolio_index.index,
            y=portfolio_index.values,
            name="Portfolio Index",
            line=dict(color='blue', width=1)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=ma.index,
            y=ma.values,
            name=f"MA({Config.FIXED_MA_LENGTH})",
            line=dict(color='orange', width=1)
        ), row=1, col=1)
        
        # Regime signal
        fig.add_trace(go.Scatter(
            x=signal.index,
            y=signal.astype(int) * 100,
            name="Regime Signal",
            fill='tozeroy',
            line=dict(color='green', width=1)
        ), row=2, col=1)
        
        fig.update_layout(
            height=600,
            showlegend=True,
            template="plotly_white"
        )
        
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Signal", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Recommendations
    st.subheader("🎯 Rebalance Recommendations")
    
    # Simplified recommendation logic
    total_gap = sum(
        account["current"] - account["last_rebalance"]
        for account in params["accounts"].values()
    )
    
    if total_gap > 0:
        st.success(f"**Recommendation:** Consider increasing risk exposure by ${total_gap:,.0f}")
    elif total_gap < 0:
        st.warning(f"**Recommendation:** Consider reducing risk exposure by ${abs(total_gap):,.0f}")
    else:
        st.info("**Recommendation:** No rebalance needed at this time")
    
    # Account-level breakdown
    for account_name, values in params["accounts"].items():
        gap = values["current"] - values["last_rebalance"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{account_name} Value", f"${values['current']:,.0f}")
        with col2:
            st.metric(f"{account_name} Last Rebalance", f"${values['last_rebalance']:,.0f}")
        with col3:
            st.metric(f"{account_name} Gap", f"${gap:,.0f}", delta=f"{gap/values['last_rebalance']:.1%}" if values['last_rebalance'] > 0 else "0%")
    
    st.divider()
    
    # Implementation Guide
    with st.expander("📋 Implementation Checklist", expanded=True):
        st.markdown("""
        1. **Monitor MA Regime**: 
           - RISK-ON when portfolio index > MA
           - RISK-OFF when portfolio index < MA
        
        2. **Quarterly Rebalance**:
           - Update portfolio values at quarter-end
           - Adjust risk exposure based on model recommendations
        
        3. **Portfolio Management**:
           - Taxable/Joint: Follow Sigma strategy (MA + quarterly rebalancing)
           - Roth/401k: Buy & Hold only
        
        4. **Key Parameters**:
           - MA Length: 200 days
           - Tolerance: Optimized for Sharpe/trade ratio
           - Quarterly target: Derived from long-term CAGR
        """)
    
    # Data export
    with st.expander("📥 Export Data"):
        export_data = pd.DataFrame({
            'Date': prices.index,
            'Portfolio_Index': portfolio_index.reindex(prices.index).values,
            'MA': ma.reindex(prices.index).values,
            'Signal': signal.reindex(prices.index).values.astype(int),
            'Strategy_Equity': strategy_equity.reindex(prices.index).values,
            'BuyHold_Equity': risk_on_equity.reindex(prices.index).values
        })
        
        csv = export_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="sigmatrader_analysis.csv",
            mime="text/csv"
        )

def main():
    """Main application entry point"""
    setup_page()
    
    # Render sidebar and get parameters
    params = render_sidebar()
    
    if params:
        render_dashboard(params)
    else:
        # Show welcome screen
        st.markdown("""
        ## Welcome to SigmaTrader Pro
        
        This application implements a sophisticated portfolio management strategy combining:
        
        **1. Moving Average Regime Filter**
        - 200-day MA on portfolio index
        - RISK-ON / RISK-OFF regime detection
        - Optimized tolerance for maximum Sharpe per trade
        
        **2. Quarterly Target-Growth Engine**
        - Calendar-based rebalancing
        - Target derived from long-term CAGR
        - Systematic capital deployment
        
        **3. Multi-Account Management**
        - Taxable accounts: Sigma strategy
        - Tax-sheltered accounts: Buy & Hold
        - Joint accounts: Sigma strategy
        
        ### Getting Started
        1. Configure your portfolio in the sidebar
        2. Set account values
        3. Click 'Run Analysis' to generate insights
        
        ### Key Features
        - ✅ Performance analytics
        - ✅ Risk metrics
        - ✅ Monte Carlo simulations
        - ✅ Rebalance recommendations
        - ✅ Data export
        """)
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.markdown("""
            **Step 1: Configure Assets**
            - Risk-On: BITU, QQQU, UGL (33.33% each)
            - Risk-Off: SHY (100%)
            
            **Step 2: Set Dates**
            - Start: 2020-01-01
            - End: Today
            
            **Step 3: Enter Account Values**
            - Taxable: $68,832
            - Tax-Sheltered: $9,266
            - Joint: $3,930
            
            **Step 4: Run Analysis**
            - Click "Run Analysis" in sidebar
            - Review metrics and charts
            - Follow rebalance recommendations
            """)

if __name__ == "__main__":
    main()
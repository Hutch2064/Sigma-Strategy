import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import json
import os
import sqlite3
import bcrypt
from datetime import datetime

# ============================================================
# SIMPLE YFINANCE LOADER (PROVEN WORKING VERSION)
# ============================================================

@st.cache_data(show_spinner=True, ttl=3600)
def load_price_data(tickers, start_date, end_date=None):
    """
    Simple yfinance download - exactly like your working old code
    """
    data = yf.download(
        tickers, 
        start=start_date, 
        end=end_date, 
        progress=False
    )

    if "Adj Close" in data.columns:
        px = data["Adj Close"].copy()
    else:
        px = data["Close"].copy()

    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])

    return px.dropna(how="all")

# ============================================================
# SQLITE DATABASE FOR USER STORAGE (SCALABLE)
# ============================================================

DB_PATH = "users_production.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_user_db():
    """Initialize SQLite database for user storage"""
    with get_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
        """)
        
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
        """)

init_user_db()

# ============================================================
# PASSWORD HASHING FUNCTIONS
# ============================================================

def hash_password_bcrypt(password: str) -> str:
    """Hash password with bcrypt for secure storage"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password_bcrypt(password: str, hashed: str) -> bool:
    """Verify password against bcrypt hash"""
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except:
        return False

# ============================================================
# USER MANAGEMENT IN SQLITE
# ============================================================

def create_user_in_db(username: str, name: str, email: str, password: str) -> bool:
    """Create new user in SQLite database"""
    try:
        with get_db() as conn:
            conn.execute(
                """INSERT INTO users (username, name, email, password_hash, last_login) 
                   VALUES (?, ?, ?, ?, ?)""",
                (username, name, email, hash_password_bcrypt(password), datetime.now())
            )
        return True
    except Exception as e:
        st.error(f"Error creating user: {str(e)}")
        return False

def get_user_from_db(username: str):
    """Get user from SQLite database"""
    with get_db() as conn:
        return conn.execute(
            """SELECT username, name, email, password_hash, is_active, last_login 
               FROM users WHERE username = ?""",
            (username,)
        ).fetchone()

def username_exists_in_db(username: str) -> bool:
    """Check if username exists in database"""
    return get_user_from_db(username) is not None

def email_exists_in_db(email: str) -> bool:
    """Check if email exists in database"""
    with get_db() as conn:
        return conn.execute(
            "SELECT 1 FROM users WHERE email = ?",
            (email,)
        ).fetchone() is not None

def update_last_login_in_db(username: str):
    """Update user's last login time"""
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE username = ?",
            (datetime.now(), username)
        )

# ============================================================
# SYNC SQLITE TO AUTHENTICATOR CONFIG
# ============================================================

def sync_db_to_authenticator():
    """
    Sync users from SQLite to streamlit-authenticator config
    This bridges SQLite storage with authenticator's cookie system
    """
    # Initialize empty config if file doesn't exist
    config = {
        "credentials": {"usernames": {}},
        "cookie": {
            "name": "sigma_portfolio_auth",
            "key": "super-secret-key-change-in-production-12345",
            "expiry_days": 30,
        },
    }
    
    # Load existing config if it exists
    if os.path.exists("auth_config.yaml"):
        with open("auth_config.yaml") as file:
            existing_config = yaml.load(file, Loader=SafeLoader)
            if existing_config:
                config = existing_config
    
    # Get all users from SQLite
    users_dict = {}
    with get_db() as conn:
        rows = conn.execute(
            "SELECT username, name, password_hash, is_active FROM users WHERE is_active = 1"
        ).fetchall()
        
        for username, name, password_hash, is_active in rows:
            if is_active:
                users_dict[username] = {
                    "name": name,
                    "password": password_hash  # Already bcrypt hashed
                }
    
    # Update config with SQLite users
    config["credentials"]["usernames"] = users_dict
    
    # Save config file
    with open("auth_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config

# ============================================================
# USER DATA PERSISTENCE
# ============================================================

USER_DATA_DIR = "user_data"
os.makedirs(USER_DATA_DIR, exist_ok=True)

DEFAULT_PREFS = {
    "start_date": "2010-01-01",
    "risk_on_tickers": "TQQQ",
    "risk_on_weights": "1.0",
    "risk_off_tickers": "AGG",
    "risk_off_weights": "1.0",
    "annual_drag_pct": 0.0,
    "qs_cap_1": 10000,
    "real_cap_1": 10000,
    "end_date": "",
    "official_inception_date": "2025-12-22",
    "benchmark_ticker": "QQQ",
    "min_holding_days": 1,
}

def _user_file(username: str) -> str:
    return os.path.join(USER_DATA_DIR, f"{username}.json")

def load_user_prefs(username: str) -> dict:
    """Load user preferences"""
    path = _user_file(username)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            pass
    return DEFAULT_PREFS.copy()

def save_user_prefs(username: str, prefs: dict):
    """Save user preferences"""
    with open(_user_file(username), "w") as f:
        json.dump(prefs, f, indent=2)

# ============================================================
# TRADING STRATEGY FUNCTIONS (FROM YOUR WORKING CODE)
# ============================================================

def show_strategy_overview():
    st.markdown("""
## **Sigma System Overview**

An automated program that integrates the SIG System with the 200 Day Simple Moving Average Strategy. 
""")

def build_portfolio_index(prices, weights_dict, annual_drag_pct=0.0):
    simple_rets = prices.pct_change().fillna(0)
    idx_rets = pd.Series(0.0, index=simple_rets.index)

    for ticker, weight in weights_dict.items():
        if ticker in simple_rets.columns:
            idx_rets += simple_rets[ticker] * weight
    
    if annual_drag_pct > 0:
        daily_drag_factor = (1 - annual_drag_pct) ** (1/252)
        idx_rets = (1 + idx_rets) * daily_drag_factor - 1
    
    cumprod = (1 + idx_rets).cumprod()
    
    valid_mask = cumprod.notna() & (cumprod > 0)
    if not valid_mask.any():
        return pd.Series(1.0, index=cumprod.index)
    
    first_valid_idx = cumprod[valid_mask].index[0]
    
    cumprod_filled = cumprod.copy()
    cumprod_filled.loc[:first_valid_idx] = 1.0
    cumprod_filled = cumprod_filled.ffill()
    
    return cumprod_filled

def compute_ma(price_series, length, ma_type):
    if ma_type.lower() == "ema":
        ma = price_series.ewm(span=length, adjust=False).mean()
    else:
        ma = price_series.rolling(window=length, min_periods=1).mean()
    
    return ma.shift(1)

def generate_testfol_signal_vectorized(price, ma, tol_series, min_holding_days=5):
    px = price.values
    ma_vals = ma.values
    n = len(px)
    
    if np.all(np.isnan(ma_vals)):
        return pd.Series(False, index=ma.index)
    
    tol_vals = tol_series.values
    upper = ma_vals * (1 + tol_vals)
    lower = ma_vals * (1 - tol_vals)
    
    sig = np.zeros(n, dtype=bool)
    
    non_nan_mask = ~np.isnan(ma_vals)
    if not np.any(non_nan_mask):
        return pd.Series(False, index=ma.index)
    
    first_valid = np.where(non_nan_mask)[0][0]
    if first_valid == 0:
        first_valid = 1
    start_index = first_valid + 1
    
    if start_index >= n:
        return pd.Series(False, index=ma.index)
    
    days_since_last_change = 0
    last_change_idx = start_index
    
    for t in range(start_index, n):
        if np.isnan(px[t]) or np.isnan(upper[t]) or np.isnan(lower[t]):
            sig[t] = sig[t-1] if t > 0 else False
        elif t - last_change_idx < min_holding_days:
            sig[t] = sig[t-1]
        elif not sig[t - 1]:
            if px[t] > upper[t]:
                sig[t] = True
                last_change_idx = t
                days_since_last_change = 0
            else:
                sig[t] = False
        else:
            if px[t] < lower[t]:
                sig[t] = False
                last_change_idx = t
                days_since_last_change = 0
            else:
                sig[t] = True
        
        if t > last_change_idx:
            days_since_last_change += 1
    
    return pd.Series(sig, index=ma.index).fillna(False)

def compute_enhanced_performance(simple_returns, eq_curve, rf=0.0):
    if len(eq_curve) == 0 or eq_curve.iloc[0] == 0:
        return {
            "CAGR": 0, "Volatility": 0, "Sharpe": 0, "MaxDrawdown": 0,
            "TotalReturn": 0, "DD_Series": pd.Series([], dtype=float),
            "Calmar": 0, "Sortino": 0, "Omega": 0
        }
    
    n_days = len(eq_curve)
    n_years = n_days / 252
    cagr = (eq_curve.iloc[-1] / eq_curve.iloc[0]) ** (1 / n_years) - 1
    vol = simple_returns.std() * np.sqrt(252) if len(simple_returns) > 0 else 0
    sharpe = (simple_returns.mean() * 252 - rf) / vol if vol > 0 else 0
    dd = eq_curve / eq_curve.cummax() - 1
    max_dd = dd.min() if len(dd) > 0 else 0
    
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "TotalReturn": eq_curve.iloc[-1] / eq_curve.iloc[0] - 1 if eq_curve.iloc[0] != 0 else 0,
    }

def backtest(prices, signal, risk_on_weights, risk_off_weights, flip_cost=0.0000):
    simple = prices.pct_change().fillna(0)
    
    # Build weights dataframe
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for a, w in risk_on_weights.items():
        if a in prices.columns:
            weights.loc[signal, a] = w

    for a, w in risk_off_weights.items():
        if a in prices.columns:
            weights.loc[~signal, a] = w
    
    strategy_simple = (weights.shift(1).fillna(0) * simple).sum(axis=1)
    eq = (1 + strategy_simple).cumprod()
    
    return {
        "returns": strategy_simple,
        "equity_curve": eq,
        "signal": signal,
        "weights": weights,
        "performance": compute_enhanced_performance(strategy_simple, eq),
    }

# ============================================================
# MAIN APP - INTEGRATED AUTHENTICATION
# ============================================================

def main():
    st.set_page_config(
        page_title="Sigma Strategy System",
        layout="wide",
        page_icon="📈"
    )
    
    # Initialize session state
    if 'authentication_status' not in st.session_state:
        st.session_state.authentication_status = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'name' not in st.session_state:
        st.session_state.name = None
    if 'prefs' not in st.session_state:
        st.session_state.prefs = None
    
    # Sync SQLite users to authenticator config
    config = sync_db_to_authenticator()
    
    # Initialize authenticator with synced config
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )
    
    # ============================================================
    # AUTHENTICATION INTERFACE
    # ============================================================
    
    # Show tabs for login/signup
    if st.session_state.authentication_status is not True:
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
        
        with login_tab:
            name, authentication_status, username = authenticator.login("Login", "main")
            
            if authentication_status is False:
                st.error("Username/password is incorrect")
            elif authentication_status is None:
                st.warning("Please enter your username and password")
            
            if authentication_status:
                # Update last login in SQLite
                update_last_login_in_db(username)
                
                # Store in session state
                st.session_state.authentication_status = True
                st.session_state.username = username
                st.session_state.name = name
                st.session_state.prefs = load_user_prefs(username)
                st.rerun()
        
        with signup_tab:
            st.subheader("Create New Account")
            
            with st.form("signup_form"):
                new_username = st.text_input("Username")
                new_name = st.text_input("Full Name")
                new_email = st.text_input("Email")
                new_password = st.text_input("Password", type="password")
                new_password_confirm = st.text_input("Confirm Password", type="password")
                
                submit = st.form_submit_button("Create Account")
                
                if submit:
                    # Validation
                    errors = []
                    if not all([new_username, new_name, new_email, new_password]):
                        errors.append("All fields are required")
                    if len(new_password) < 8:
                        errors.append("Password must be at least 8 characters")
                    if new_password != new_password_confirm:
                        errors.append("Passwords do not match")
                    if username_exists_in_db(new_username):
                        errors.append("Username already exists")
                    if email_exists_in_db(new_email):
                        errors.append("Email already registered")
                    
                    if errors:
                        for error in errors:
                            st.error(error)
                    else:
                        # Create user in SQLite
                        if create_user_in_db(new_username, new_name, new_email, new_password):
                            # Sync to authenticator config
                            sync_db_to_authenticator()
                            st.success("✅ Account created successfully! Please log in.")
                        else:
                            st.error("Failed to create account")
        
        st.stop()  # Stop here if not authenticated
    
    # ============================================================
    # AUTHENTICATED USER INTERFACE
    # ============================================================
    
    # User is authenticated at this point
    st.sidebar.title(f"Welcome {st.session_state.name}!")
    
    # Logout button
    authenticator.logout("Logout", "sidebar")
    
    # Main app content
    show_strategy_overview()
    st.markdown("---")
    
    # Load user preferences
    if st.session_state.prefs is None:
        st.session_state.prefs = load_user_prefs(st.session_state.username)
    
    prefs = st.session_state.prefs
    
    # ============================================================
    # STRATEGY SETTINGS
    # ============================================================
    
    st.sidebar.header("⚙️ Strategy Settings")
    
    with st.sidebar.expander("Portfolio Configuration", expanded=True):
        start = st.text_input("Start Date", prefs["start_date"])
        risk_on_tickers_str = st.text_input("Risk On Tickers", prefs["risk_on_tickers"])
        risk_on_weights_str = st.text_input("Risk On Weights", prefs["risk_on_weights"])
        risk_off_tickers_str = st.text_input("Risk Off Tickers", prefs["risk_off_tickers"])
        risk_off_weights_str = st.text_input("Risk Off Weights", prefs["risk_off_weights"])
        annual_drag = st.number_input("Annual Drag %", value=float(prefs["annual_drag_pct"]), min_value=0.0, max_value=100.0, step=0.1)
    
    with st.sidebar.expander("Portfolio Values", expanded=True):
        qs_cap_1 = st.number_input("Portfolio Value at Last Rebalance", value=float(prefs["qs_cap_1"]), min_value=0.0, step=1000.0)
        real_cap_1 = st.number_input("Portfolio Value Today", value=float(prefs["real_cap_1"]), min_value=0.0, step=1000.0)
    
    with st.sidebar.expander("Advanced Settings", expanded=False):
        inception_date = st.text_input("Inception Date", prefs["official_inception_date"])
        benchmark = st.text_input("Benchmark", prefs["benchmark_ticker"])
        min_days = st.number_input("Confirmation Days", value=int(prefs["min_holding_days"]), min_value=1, max_value=30)
    
    # Save settings button
    if st.sidebar.button("💾 Save Settings", type="secondary", use_container_width=True):
        st.session_state.prefs = {
            "start_date": start,
            "risk_on_tickers": risk_on_tickers_str,
            "risk_on_weights": risk_on_weights_str,
            "risk_off_tickers": risk_off_tickers_str,
            "risk_off_weights": risk_off_weights_str,
            "annual_drag_pct": annual_drag,
            "qs_cap_1": qs_cap_1,
            "real_cap_1": real_cap_1,
            "end_date": "",
            "official_inception_date": inception_date,
            "benchmark_ticker": benchmark,
            "min_holding_days": min_days,
        }
        save_user_prefs(st.session_state.username, st.session_state.prefs)
        st.sidebar.success("Settings saved!")
    
    # Run analysis button
    run_clicked = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    if not run_clicked:
        st.info("Adjust settings in sidebar and click 'Run Analysis' to begin.")
        return
    
    # ============================================================
    # STRATEGY EXECUTION
    # ============================================================
    
    try:
        # Process inputs
        risk_on_tickers = [t.strip().upper() for t in risk_on_tickers_str.split(",") if t.strip()]
        risk_on_weights_list = [float(x.strip()) for x in risk_on_weights_str.split(",") if x.strip()]
        
        if len(risk_on_tickers) != len(risk_on_weights_list):
            st.error(f"Mismatch: {len(risk_on_tickers)} tickers but {len(risk_on_weights_list)} weights")
            return
        
        risk_on_weights = dict(zip(risk_on_tickers, risk_on_weights_list))
        
        risk_off_tickers = [t.strip().upper() for t in risk_off_tickers_str.split(",") if t.strip()]
        risk_off_weights_list = [float(x.strip()) for x in risk_off_weights_str.split(",") if x.strip()]
        
        if len(risk_off_tickers) != len(risk_off_weights_list):
            st.error(f"Mismatch: {len(risk_off_tickers)} tickers but {len(risk_off_weights_list)} weights")
            return
        
        risk_off_weights = dict(zip(risk_off_tickers, risk_off_weights_list))
        
        annual_drag_decimal = annual_drag / 100.0
        
        # Combine all tickers
        all_tickers = list(set(risk_on_tickers + risk_off_tickers))
        
        # Load price data with simple yfinance call
        with st.spinner(f"Loading price data for {', '.join(all_tickers)}..."):
            prices = load_price_data(all_tickers, start)
        
        if prices.empty:
            st.error("No data loaded. Please check your ticker symbols and date range.")
            return
        
        st.success(f"✅ Data loaded from {prices.index[0].date()} to {prices.index[-1].date()}")
        
        # MA setup
        best_len = 200
        best_type = "sma"
        
        portfolio_index = build_portfolio_index(prices, risk_on_weights, annual_drag_pct=annual_drag_decimal)
        opt_ma = compute_ma(portfolio_index, best_len, best_type)
        
        tolerance_decimal = 0.0
        tol_series = pd.Series(tolerance_decimal, index=portfolio_index.index)
        
        sig = generate_testfol_signal_vectorized(
            portfolio_index,
            opt_ma,
            tol_series,
            min_holding_days=min_days
        )
        
        # Run backtest
        best_result = backtest(prices, sig, risk_on_weights, risk_off_weights)
        
        latest_signal = sig.iloc[-1] if not sig.empty else False
        current_regime = "Risk On" if latest_signal else "Risk Off"
        
        # Get returns data
        simple_rets = prices.pct_change().fillna(0)
        risk_on_simple = pd.Series(0.0, index=simple_rets.index)
        for a, w in risk_on_weights.items():
            if a in simple_rets.columns:
                risk_on_simple += simple_rets[a] * w
        
        if annual_drag_decimal > 0:
            daily_drag_factor = (1 - annual_drag_decimal) ** (1/252)
            risk_on_simple = (1 + risk_on_simple) * daily_drag_factor - 1
        
        risk_on_eq = (1 + risk_on_simple).cumprod()
        
        # Since-inception analysis
        inception = pd.to_datetime(inception_date)
        ma_eq_si = best_result["equity_curve"].loc[best_result["equity_curve"].index >= inception]
        ma_ret_si = ma_eq_si.pct_change().fillna(0)
        
        bh_eq_si = risk_on_eq.loc[risk_on_eq.index >= inception]
        bh_ret_si = bh_eq_si.pct_change().fillna(0)
        
        # Load benchmark
        benchmark_px = load_price_data([benchmark], inception)
        if not benchmark_px.empty and benchmark in benchmark_px.columns:
            benchmark_eq_si = (benchmark_px[benchmark] / benchmark_px[benchmark].iloc[0]).reindex(ma_eq_si.index).ffill()
            benchmark_ret_si = benchmark_eq_si.pct_change().fillna(0)
        else:
            benchmark_eq_si = pd.Series(1.0, index=ma_eq_si.index)
            benchmark_ret_si = pd.Series(0.0, index=ma_eq_si.index)
        
        # ============================================================
        # DISPLAY RESULTS
        # ============================================================
        
        st.subheader("📊 Performance Results")
        
        # Current regime
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current 200-Day SMA Regime", current_regime)
        with col2:
            st.metric("Portfolio Value Today", f"${real_cap_1:,.2f}")
        with col3:
            # Calculate quarterly target
            if len(risk_on_eq) > 0 and risk_on_eq.iloc[0] != 0:
                bh_cagr = (risk_on_eq.iloc[-1] / risk_on_eq.iloc[0]) ** (252 / len(risk_on_eq)) - 1
                quarterly_target = (1 + bh_cagr) ** (1/4) - 1
                st.metric("Quarterly Target", f"{quarterly_target:.2%}")
            else:
                st.metric("Quarterly Target", "0.00%")
        
        # Performance comparison chart
        st.subheader(f"Performance (200-Day MA vs Buy & Hold vs {benchmark})")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(ma_eq_si / ma_eq_si.iloc[0], label="200-Day MA", linewidth=2, color="blue")
        ax.plot(bh_eq_si / bh_eq_si.iloc[0], label="Buy & Hold", linewidth=2, alpha=0.7)
        ax.plot(benchmark_eq_si, label=benchmark, linewidth=2, linestyle="--", color="black", alpha=0.7)
        ax.set_ylabel("Growth of $1")
        ax.set_title(f"Performance Since {inception_date}")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        # Performance metrics
        ma_perf_si = compute_enhanced_performance(ma_ret_si, ma_eq_si)
        bh_perf_si = compute_enhanced_performance(bh_ret_si, bh_eq_si)
        benchmark_perf_si = compute_enhanced_performance(benchmark_ret_si, benchmark_eq_si)
        
        def fmt(val, kind):
            if pd.isna(val):
                return "—"
            if kind == "pct":
                return f"{val:.2%}"
            return f"{val:.3f}"
        
        metrics_data = []
        metrics = [
            ("CAGR", "CAGR", "pct"),
            ("Volatility", "Volatility", "pct"),
            ("Sharpe", "Sharpe", "dec"),
            ("Max Drawdown", "MaxDrawdown", "pct"),
            ("Total Return", "TotalReturn", "pct"),
        ]
        
        for label, key, kind in metrics:
            metrics_data.append([
                label,
                fmt(ma_perf_si[key], kind),
                fmt(bh_perf_si[key], kind),
                fmt(benchmark_perf_si[key], kind),
            ])
        
        metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "200-Day MA", "Buy & Hold", benchmark])
        st.dataframe(metrics_df, use_container_width=True)
        
        # MA distance
        st.subheader("📐 200-Day SMA Crossover Distance")
        if len(opt_ma) > 0 and len(portfolio_index) > 0:
            latest_date = opt_ma.dropna().index[-1] if not opt_ma.dropna().empty else None
            if latest_date and latest_date in portfolio_index.index:
                P = float(portfolio_index.loc[latest_date])
                MA = float(opt_ma.loc[latest_date])
                upper = MA * (1 + tolerance_decimal)
                lower = MA * (1 - tolerance_decimal)
                
                if latest_signal:
                    delta = (P - lower) / P if P > 0 else 0
                    st.info(f"**Drop Required for Crossover:** {delta:.2%}")
                else:
                    delta = (upper - P) / P if P > 0 else 0
                    st.info(f"**Gain Required for Crossover:** {delta:.2%}")
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.info("Please check your inputs and try again.")

# ============================================================
# RUN THE APP
# ============================================================

if __name__ == "__main__":
    main()

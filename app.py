import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import bcrypt
import json
import os
import sqlite3
import secrets
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

# ============================================================
# DATABASE SETUP (Production Ready)
# ============================================================

DB_PATH = "users.db"
USER_DATA_DIR = "user_data"
os.makedirs(USER_DATA_DIR, exist_ok=True)

def get_db():
    """Get database connection with proper isolation"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn

def init_auth_db():
    """Initialize database with proper indexes"""
    with get_db() as conn:
        # Users table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email_verified INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
        """)
        
        # Password resets
        conn.execute("""
        CREATE TABLE IF NOT EXISTS password_resets (
            token_hash TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        )
        """)
        
        # Login attempts (for rate limiting)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS login_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            ip_address TEXT,
            attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            success INTEGER DEFAULT 0
        )
        """)
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_password_resets_expires ON password_resets(expires_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_login_attempts_time ON login_attempts(attempt_time)")
        
init_auth_db()

# ============================================================
# SECURITY & AUTHENTICATION FUNCTIONS
# ============================================================

def hash_password(password: str) -> str:
    """Hash password with bcrypt"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except:
        return False

def get_user(username: str):
    """Get user by username"""
    with get_db() as conn:
        return conn.execute(
            """SELECT username, name, email, password_hash, email_verified, 
                      is_active, created_at, last_login 
               FROM users WHERE username = ?""",
            (username,)
        ).fetchone()

def username_exists(username: str) -> bool:
    """Check if username exists"""
    return get_user(username) is not None

def email_exists(email: str) -> bool:
    """Check if email exists"""
    with get_db() as conn:
        return conn.execute(
            "SELECT 1 FROM users WHERE email = ?",
            (email,)
        ).fetchone() is not None

def create_user(username: str, name: str, email: str, password: str):
    """Create new user"""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO users (username, name, email, password_hash) 
               VALUES (?, ?, ?, ?)""",
            (username, name, email, hash_password(password))
        )

def update_last_login(username: str):
    """Update user's last login time"""
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = ?",
            (username,)
        )

def create_token(username: str, minutes: int = 30) -> str:
    """Create verification/reset token"""
    raw = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    expires = datetime.utcnow() + timedelta(minutes=minutes)

    with get_db() as conn:
        # Clean old tokens
        conn.execute(
            "DELETE FROM password_resets WHERE username = ? OR expires_at < ?",
            (username, datetime.utcnow())
        )
        conn.execute(
            """INSERT INTO password_resets (token_hash, username, expires_at) 
               VALUES (?, ?, ?)""",
            (token_hash, username, expires)
        )
    return raw

def validate_token(raw_token: str):
    """Validate token and return username if valid"""
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    now = datetime.utcnow()

    with get_db() as conn:
        row = conn.execute(
            """SELECT username FROM password_resets 
               WHERE token_hash = ? AND expires_at > ?""",
            (token_hash, now)
        ).fetchone()
    return row[0] if row else None

def consume_token(username: str):
    """Remove used token"""
    with get_db() as conn:
        conn.execute(
            "DELETE FROM password_resets WHERE username = ?",
            (username,)
        )

def send_email_simple(to_email: str, subject: str, html_body: str):
    """Simple email sending function (replace with your email service)"""
    try:
        # Using environment variables for email configuration
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        from_email = os.getenv("FROM_EMAIL", "noreply@yourdomain.com")
        
        if not all([smtp_user, smtp_password]):
            st.warning("Email not configured. Please set SMTP environment variables.")
            return False
            
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email

        # Plain-text fallback (VERY important for Gmail)
        text_part = MIMEText(
            "This email was sent to verify your Sigma System account. "
            "If you did not create an account, you can ignore this email.",
            "plain",
        )

        # HTML version (existing content)
        html_part = MIMEText(html_body, "html")

        msg.attach(text_part)
        msg.attach(html_part)
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email sending failed: {str(e)}")
        return False

# ============================================================
# USER DATA PERSISTENCE
# ============================================================

DEFAULT_PREFS = {
    "start_date": "1900-01-01",
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
# TRADING STRATEGY FUNCTIONS (From your original code)
# ============================================================

def show_strategy_overview():
    st.markdown("""
## **Sigma System Overview**

An automated program that integrates the SIG System with the 200 Day Simple Moving Average Strategy. 

---

### **The SIG System:**

**Quarterly Target Growth Rate (QTGR):**  
Quarterly Growth Target (QGT): Quarterly growth rate derived from the historical returns of the user selected Risk On allocation (e.g., 9sig = 9% QGT for TQQQ).

- Is your Risk On Allocation above or below its quarterly growth target? If above, it's a sell signal. If below, it's a buy signal.
- If the signal is a Risk On Allocation sell, you will move proceeds of the sale to your Risk Off Allocation in the following order of events: sell an amount of the Risk On Allocation, use that amount to buy more of the Risk Off Allocation. If the signal is a Risk On Allocation buy, you will generate buying power by selling a portion of your Risk Off Allocation, then use the proceeds to buy more of your Risk On Allocation, in the following order of events: sell a portion of the Risk Off Allocation, use the proceeds to buy the Risk On Allocation.
---

### **The 200 Day Simple Moving Average Strategy (SMA or MA):**

A 200 Day SMA is constructed using a simulated index of the user selected Risk On allocation.

- If the 200 Day SMA < Risk On Allocation Index, then buy & hold the Risk On Allocation.
- If the 200 Day SMA > Risk On Allocation Index, then sell & hold the Risk Off Allocation.

A "Risk On Regime" = 200 Day SMA < Risk On Allocation Index.  
A "Risk Off Regime" = 200 Day SMA > Risk On Allocation Index.

---

### **Sigma System**

- If the 200 Day SMA < Risk On Allocation Index, then the model runs the SIG System as instructed above.
- If the 200 Day SMA > Risk On Allocation Index, then the model allocates all portfolio capital to the Risk Off Allocation.
- When model flips from "Risk Off" to "Risk On", the model refers to the Allocation Tables and resumes the current SIG System weights.
""")

# CONFIG
DEFAULT_START_DATE = "1900-01-01"
RISK_FREE_RATE = 0.0
RISK_ON_WEIGHTS = {"TQQQ": 1.0}
RISK_OFF_WEIGHTS = {"AGG": 1.0}
FLIP_COST = 0.0000
START_RISKY = 0.6
START_SAFE  = 0.4

@st.cache_data(show_spinner=True, ttl=3600)
def load_price_data(tickers, start_date, end_date=None):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if "Adj Close" in data.columns:
        px = data["Adj Close"].copy()
        if "Close" in data.columns:
            px = px.combine_first(data["Close"])
    else:
        px = data["Close"].copy()

    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])

    return px.dropna(how="all")

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

def run_sig_engine(
    risk_on_returns,
    risk_off_returns,
    target_quarter,
    ma_signal,
    pure_sig_rw=None,
    pure_sig_sw=None,
    flip_cost=FLIP_COST,
    quarter_end_dates=None,
    quarterly_multiplier=4.0,
    ma_flip_multiplier=4.0
):

    dates = risk_on_returns.index
    n = len(dates)

    if quarter_end_dates is None:
        raise ValueError("quarter_end_dates must be supplied")

    quarter_end_set = set(quarter_end_dates)
    sig_arr = ma_signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1

    eq = 10000.0
    risky_val = eq * START_RISKY
    safe_val  = eq * START_SAFE

    frozen_risky = None
    frozen_safe  = None

    equity_curve = []
    risky_w_series = []
    safe_w_series = []
    risky_val_series = []
    safe_val_series = []
    rebalance_events = 0
    rebalance_dates = []

    for i in range(n):
        date = dates[i]
        r_on = risk_on_returns.iloc[i]
        r_off = risk_off_returns.iloc[i]
        ma_on = bool(ma_signal.iloc[i])
        
        if i > 0 and flip_mask.iloc[i]:
            eq *= (1 - flip_cost * ma_flip_multiplier)

        if ma_on:
            if frozen_risky is not None:
                w_r = pure_sig_rw.iloc[i]
                w_s = pure_sig_sw.iloc[i]
                risky_val = eq * w_r
                safe_val  = eq * w_s
                frozen_risky = None
                frozen_safe  = None

            risky_val *= (1 + r_on)
            safe_val  *= (1 + r_off)

            if date in quarter_end_set:
                prev_qs = [qd for qd in quarter_end_dates if qd < date]

                if prev_qs:
                    prev_q = prev_qs[-1]
                    idx_prev = dates.get_loc(prev_q)
                    risky_at_qstart = risky_val_series[idx_prev]
                    goal_risky = risky_at_qstart * (1 + target_quarter)

                    if risky_val > goal_risky:
                        excess = risky_val - goal_risky
                        risky_val -= excess
                        safe_val  += excess
                        rebalance_dates.append(date)

                    elif risky_val < goal_risky:
                        needed = goal_risky - risky_val
                        move = min(needed, safe_val)
                        safe_val -= move
                        risky_val += move
                        rebalance_dates.append(date)

                    eq *= (1 - flip_cost * quarterly_multiplier)

            eq = risky_val + safe_val
            risky_w = risky_val / eq
            safe_w  = safe_val  / eq

        else:
            if frozen_risky is None:
                frozen_risky = risky_val
                frozen_safe  = safe_val

            eq *= (1 + r_off)
            risky_w = 0.0
            safe_w  = 1.0

        equity_curve.append(eq)
        risky_w_series.append(risky_w)
        safe_w_series.append(safe_w)
        risky_val_series.append(risky_val)
        safe_val_series.append(safe_val)

    return (
        pd.Series(equity_curve, index=dates),
        pd.Series(risky_w_series, index=dates),
        pd.Series(safe_w_series, index=dates),
        rebalance_dates
    )

def build_weight_df(prices, signal, risk_on_weights, risk_off_weights):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for a, w in risk_on_weights.items():
        if a in prices.columns:
            weights.loc[signal, a] = w

    for a, w in risk_off_weights.items():
        if a in prices.columns:
            weights.loc[~signal, a] = w

    return weights

def compute_enhanced_performance(simple_returns, eq_curve, rf=0.0):
    if len(eq_curve) == 0 or eq_curve.iloc[0] == 0:
        return {
            "CAGR": 0, "Volatility": 0, "Sharpe": 0, "MaxDrawdown": 0,
            "TotalReturn": 0, "DD_Series": pd.Series([], dtype=float),
            "Calmar": 0, "Sortino": 0, "Omega": 0, "Skewness": 0,
            "Kurtosis": 0, "VaR_95": 0, "CVaR_95": 0, "WinRate": 0,
            "ProfitFactor": 0, "RecoveryFactor": 0, "UlcerIndex": 0, "TailRatio": 0
        }
    
    n_days = len(eq_curve)
    n_years = n_days / 252
    cagr = (eq_curve.iloc[-1] / eq_curve.iloc[0]) ** (1 / n_years) - 1
    vol = simple_returns.std() * np.sqrt(252) if len(simple_returns) > 0 else 0
    sharpe = (simple_returns.mean() * 252 - rf) / vol if vol > 0 else 0
    dd = eq_curve / eq_curve.cummax() - 1
    max_dd = dd.min() if len(dd) > 0 else 0
    
    downside_returns = simple_returns[simple_returns < 0]
    downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = (simple_returns.mean() * 252 - rf) / downside_dev if downside_dev > 0 else 0
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    threshold = 0.0
    gains = simple_returns[simple_returns > threshold].sum()
    losses = abs(simple_returns[simple_returns < threshold].sum())
    omega = gains / losses if losses > 0 else 0
    
    positive_rets = simple_returns[simple_returns > 0]
    negative_rets = simple_returns[simple_returns < 0]
    win_rate = len(positive_rets) / len(simple_returns) if len(simple_returns) > 0 else 0
    gross_profit = positive_rets.sum()
    gross_loss = abs(negative_rets.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    var_95 = np.percentile(simple_returns, 5) * np.sqrt(252)
    cvar_95 = simple_returns[simple_returns <= np.percentile(simple_returns, 5)].mean() * np.sqrt(252) if len(simple_returns) > 0 else 0
    
    skewness = simple_returns.skew() if len(simple_returns) > 0 else 0
    kurtosis = simple_returns.kurt() if len(simple_returns) > 0 else 0
    
    ulcer_index = np.sqrt((dd ** 2).mean()) if len(dd) > 0 else 0
    
    recovery_factor = -cagr / max_dd if max_dd != 0 else 0
    
    top_5 = np.percentile(simple_returns, 95)
    bottom_5 = np.percentile(simple_returns, 5)
    tail_ratio = abs(top_5 / bottom_5) if bottom_5 != 0 else 0
    
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Omega": omega,
        "MaxDrawdown": max_dd,
        "TotalReturn": eq_curve.iloc[-1] / eq_curve.iloc[0] - 1 if eq_curve.iloc[0] != 0 else 0,
        "WinRate": win_rate,
        "ProfitFactor": profit_factor,
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "UlcerIndex": ulcer_index,
        "RecoveryFactor": recovery_factor,
        "TailRatio": tail_ratio,
        "DD_Series": dd
    }

def backtest(prices, signal, risk_on_weights, risk_off_weights, flip_cost, ma_flip_multiplier=3.0, annual_drag_pct=0.0):
    simple = prices.pct_change().fillna(0)
    weights = build_weight_df(prices, signal, risk_on_weights, risk_off_weights)

    strategy_simple = (weights.shift(1).fillna(0) * simple).sum(axis=1)
    sig_arr = signal.astype(int)
    flip_mask = sig_arr.diff().abs() == 1

    flip_costs = np.where(flip_mask, -flip_cost * ma_flip_multiplier, 0.0)
    
    if annual_drag_pct > 0:
        daily_drag_factor = (1 - annual_drag_pct) ** (1/252)
        strategy_simple = (1 + strategy_simple) * daily_drag_factor - 1
    
    strat_adj = strategy_simple + flip_costs

    eq = (1 + strat_adj).cumprod()

    return {
        "returns": strat_adj,
        "equity_curve": eq,
        "signal": signal,
        "weights": weights,
        "performance": compute_enhanced_performance(strat_adj, eq),
        "flip_mask": flip_mask,
    }

def compute_quarter_progress(risky_start, risky_today, quarterly_target):
    target_risky = risky_start * (1 + quarterly_target)
    gap = target_risky - risky_today
    pct_gap = gap / risky_start if risky_start > 0 else 0

    return {
        "Risk On Capital at Last Rebalance ($)": risky_start,
        "Risk On Capital Today ($)": risky_today,
        "Risk On Capital Target Next Rebalance ($)": target_risky,
        "Gap ($)": gap,
        "Gap (%)": pct_gap,
    }

def normalize(eq):
    if len(eq) == 0 or eq.iloc[0] == 0:
        return eq
    return eq / eq.iloc[0] * 10000

def plot_diagnostics(hybrid_eq, bh_eq, hybrid_signal):
    hybrid_eq = hybrid_eq / hybrid_eq.iloc[0]
    bh_eq     = bh_eq / bh_eq.iloc[0]

    hybrid_ret = hybrid_eq.pct_change().fillna(0)
    bh_ret = bh_eq.pct_change().fillna(0)

    hybrid_dd = hybrid_eq / hybrid_eq.cummax() - 1
    bh_dd = bh_eq / bh_eq.cummax() - 1

    window = 252
    roll_sharpe_h = hybrid_ret.rolling(window).mean() / hybrid_ret.rolling(window).std() * np.sqrt(252)
    roll_sharpe_b = bh_ret.rolling(window).mean() / bh_ret.rolling(window).std() * np.sqrt(252)

    hybrid_m = hybrid_ret.resample("M").apply(lambda x: (1 + x).prod() - 1)
    bh_m = bh_ret.resample("M").apply(lambda x: (1 + x).prod() - 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    ax1.plot(hybrid_eq, label="Sigma", linewidth=2, color="green")
    ax1.plot(bh_eq, label="Buy & Hold", linewidth=2, alpha=0.7)

    in_off = False
    start = None
    for date, on in hybrid_signal.items():
        if not on and not in_off:
            start = date
            in_off = True
        elif on and in_off:
            ax1.axvspan(start, date, color="red", alpha=0.15)
            in_off = False
    if in_off:
        ax1.axvspan(start, hybrid_signal.index[-1], color="red", alpha=0.15)

    ax1.set_title("Cumulative Returns with Regime Shading")
    ax1.set_ylabel("Growth of $1")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(hybrid_dd * 100, label="Sigma", linewidth=1.5, color="green")
    ax2.plot(bh_dd * 100, label="Buy & Hold", linewidth=1.5, alpha=0.7)
    ax2.set_title("Drawdown Comparison (%)")
    ax2.set_ylabel("Drawdown %")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3.plot(roll_sharpe_h, label="Sigma", linewidth=1.5, color="green")
    ax3.plot(roll_sharpe_b, label="Buy & Hold", linewidth=1.5, alpha=0.7)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_title("Rolling 252-Day Sharpe Ratio")
    ax3.legend()
    ax3.grid(alpha=0.3)

    bins = np.linspace(
        min(hybrid_m.min(), bh_m.min()),
        max(hybrid_m.max(), bh_m.max()),
        20
    )

    ax4.hist(hybrid_m, bins=bins, alpha=0.7, density=True, label="Sigma")
    ax4.hist(bh_m, bins=bins, alpha=0.5, density=True, label="Buy & Hold")
    ax4.axvline(0, color="black", linestyle="--", linewidth=1)
    ax4.set_title("Monthly Returns Distribution")
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    return fig

def monte_carlo_strategy_analysis(strategy_returns, strategy_equity, n_sim=10000, periods=252, initial_capital=None):
    if len(strategy_returns) < 100:
        return None
    
    mu_daily = strategy_returns.mean()
    sigma_daily = strategy_returns.std()
    
    if initial_capital is not None:
        initial_price = initial_capital
    else:
        initial_price = strategy_equity.iloc[-1] if len(strategy_equity) > 0 else 10000
    
    np.random.seed(42)
    sim_returns = np.random.normal(mu_daily, sigma_daily, (n_sim, periods))
    
    sim_values = initial_price * np.cumprod(1 + sim_returns, axis=1)
    
    terminal_values = sim_values[:, -1]
    terminal_returns = (terminal_values / initial_price) - 1
    
    percentiles = np.percentile(terminal_returns, list(range(5, 100, 5)))
    
    def calculate_cvar(returns, confidence):
        threshold = np.percentile(returns, 100 - confidence)
        bad_returns = returns[returns <= threshold]
        return -np.mean(bad_returns) if len(bad_returns) > 0 else 0
    
    cvar_90 = calculate_cvar(terminal_returns, 90)
    cvar_95 = calculate_cvar(terminal_returns, 95)
    cvar_99 = calculate_cvar(terminal_returns, 99)
    
    expected_return = np.mean(terminal_returns)
    expected_vol = np.std(terminal_returns)
    prob_positive = np.mean(terminal_returns > 0)
    
    terminal_value_percentiles = np.percentile(terminal_values, [5, 25, 50, 75, 95])
    
    return {
        'sim_prices': sim_values,
        'terminal_values': terminal_values,
        'terminal_returns': terminal_returns,
        'percentiles': percentiles,
        'terminal_value_percentiles': terminal_value_percentiles,
        'cvar_90': cvar_90,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'expected_return': expected_return,
        'expected_vol': expected_vol,
        'prob_positive': prob_positive,
        'var_95': np.percentile(terminal_returns, 5),
        'var_99': np.percentile(terminal_returns, 1),
        'initial_price': initial_price
    }

def plot_monte_carlo_results(results_dict, strategy_names):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))
    
    ax = axes[0, 0]
    for i, (name, results) in enumerate(results_dict.items()):
        if results is not None:
            returns_pct = results['terminal_returns'] * 100
            ax.hist(returns_pct, bins=50, alpha=0.5, 
                   label=name, density=True, color=colors[i])
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_title('12-Month Return Distributions')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    percentile_levels = list(range(5, 100, 5))
    for i, (name, results) in enumerate(results_dict.items()):
        if results is not None:
            percentiles_pct = results['percentiles'] * 100
            ax.plot(percentile_levels, percentiles_pct, 
                   marker='o', label=name, color=colors[i])
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Percentile Return Ranges (5%-95%)')
    ax.set_xlabel('Percentile')
    ax.set_ylabel('12-Month Return (%)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 2]
    cvar_data = []
    labels = []
    for name, results in results_dict.items():
        if results is not None:
            cvar_data.append([results['cvar_95'] * 100, results['cvar_99'] * 100])
            labels.append(name)
    
    if cvar_data:
        cvar_data = np.array(cvar_data)
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, cvar_data[:, 0], width, label='CVaR 95%', alpha=0.7)
        ax.bar(x + width/2, cvar_data[:, 1], width, label='CVaR 99%', alpha=0.7)
        ax.set_title('Conditional Value at Risk (CVaR)')
        ax.set_ylabel('CVaR (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    for i, (name, results) in enumerate(results_dict.items()):
        if results is not None:
            for j in range(min(20, results['sim_prices'].shape[0])):
                ax.plot(results['sim_prices'][j, :], alpha=0.1, color=colors[i])
    ax.set_title('Sample Portfolio Paths ($)')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    for i, (name, results) in enumerate(results_dict.items()):
        if results is not None:
            ax.scatter(results['expected_vol'], results['expected_return'] * 100, 
                      s=100, label=name, color=colors[i], alpha=0.7)
            ax.text(results['expected_vol']*1.01, results['expected_return']*100*1.01, 
                   name, fontsize=9, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_title('Expected Risk-Return Profile (Annualized)')
    ax.set_xlabel('Expected Volatility (Annualized)')
    ax.set_ylabel('Expected Return (% Annualized)')
    ax.grid(alpha=0.3)
    
    ax = axes[1, 2]
    prob_data = []
    prob_labels = []
    for name, results in results_dict.items():
        if results is not None:
            prob_data.append(results['prob_positive'] * 100)
            prob_labels.append(name)
    
    if prob_data:
        bars = ax.bar(range(len(prob_data)), prob_data, color=colors[:len(prob_data)])
        ax.set_title('Probability of Positive 12-Month Return')
        ax.set_ylabel('Probability (%)')
        ax.set_xticks(range(len(prob_data)))
        ax.set_xticklabels(prob_labels, rotation=45, ha='right')
        ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        for bar, val in zip(bars, prob_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================
# STREAMLIT APP - MAIN FUNCTION
# ============================================================

def main():
    st.set_page_config(
        page_title="Portfolio MA Regime Strategy",
        layout="wide",
        page_icon="📈"
    )
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "name" not in st.session_state:
        st.session_state.name = None
    
    # ============================================================
    # EMAIL VERIFICATION & PASSWORD RESET HANDLING
    # ============================================================
    
    params = st.query_params
    
    # Email verification
    if "verify" in params:
        st.title("Email Verification")
        username = validate_token(params["verify"])
        if username:
            with get_db() as conn:
                conn.execute(
                    "UPDATE users SET email_verified = 1 WHERE username = ?",
                    (username,)
                )
            consume_token(username)
            st.success("✅ Email verified successfully! You can now log in.")
            st.markdown("[Click here to login](/)" if st.secrets.get("APP_URL") else "Return to the login page")
        else:
            st.error("❌ Verification link is invalid or expired.")
        st.stop()
    
    # Password reset
    if "reset" in params:
        st.title("Reset Password")
        reset_user = validate_token(params["reset"])
        if not reset_user:
            st.error("❌ Reset link is invalid or expired.")
            st.stop()
        
        with st.form("reset_password_form"):
            st.write(f"Setting new password for user: **{reset_user}**")
            pw1 = st.text_input("New Password", type="password", key="pw1")
            pw2 = st.text_input("Confirm Password", type="password", key="pw2")
            submit = st.form_submit_button("Reset Password")
            
            if submit:
                if not pw1 or pw1 != pw2:
                    st.error("Passwords do not match.")
                elif len(pw1) < 8:
                    st.error("Password must be at least 8 characters.")
                else:
                    with get_db() as conn:
                        conn.execute(
                            "UPDATE users SET password_hash = ? WHERE username = ?",
                            (hash_password(pw1), reset_user)
                        )
                    consume_token(reset_user)
                    st.success("✅ Password reset successfully! You may now log in.")
                    if st.button("Go to Login"):
                        st.query_params.clear()
                        st.rerun()
        st.stop()
    
    # ============================================================
    # AUTHENTICATION GATE
    # ============================================================
    
    if not st.session_state.authenticated:
        # Show login/signup interface
        tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Reset Password"])
        
        with tab1:
            st.subheader("Login")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                
                if submit:
                    if not username or not password:
                        st.error("Please enter username and password")
                    else:
                        user_data = get_user(username)
                        if user_data:
                            if user_data[5] == 0:  # is_active check
                                st.error("Account is disabled. Please contact support.")
                            elif user_data[4] == 0:  # email_verified check
                                st.error("Email not verified. Please check your email.")
                            elif verify_password(password, user_data[3]):
                                # Successful login
                                update_last_login(username)
                                st.session_state.update({
                                    "authenticated": True,
                                    "username": username,
                                    "name": user_data[1],
                                    "prefs": load_user_prefs(username)
                                })
                                st.success(f"Welcome back, {user_data[1]}!")
                                st.rerun()
                            else:
                                st.error("Invalid username or password")
                        else:
                            st.error("Invalid username or password")
        
        with tab2:
            st.subheader("Create New Account")
            with st.form("signup_form"):
                new_username = st.text_input("Username", help="Choose a unique username")
                new_name = st.text_input("Full Name")
                new_email = st.text_input("Email")
                new_pw = st.text_input("Password", type="password", 
                                      help="At least 8 characters")
                new_pw2 = st.text_input("Confirm Password", type="password")
                submit = st.form_submit_button("Create Account")
                
                if submit:
                    # Validation
                    errors = []
                    if not all([new_username, new_name, new_email, new_pw]):
                        errors.append("All fields are required")
                    if len(new_pw) < 8:
                        errors.append("Password must be at least 8 characters")
                    if new_pw != new_pw2:
                        errors.append("Passwords do not match")
                    if username_exists(new_username):
                        errors.append("Username already exists")
                    if email_exists(new_email):
                        errors.append("Email already registered")
                    if "@" not in new_email or "." not in new_email:
                        errors.append("Invalid email format")
                    
                    if errors:
                        for error in errors:
                            st.error(error)
                    else:
                        try:
                            create_user(new_username, new_name, new_email, new_pw)
                            token = create_token(new_username, minutes=60)
                            app_url = st.secrets.get("APP_URL", "http://localhost:8501")
                            verify_link = f"{app_url}?verify={token}"
                            
                            # Send verification email
                            email_sent = send_email_simple(
                                            new_email,
                                            "Verify Your Email - Portfolio Strategy App",
                                            f"""
                                            <p style="font-size:12px;color:#666;">
                                                This is a transactional email to verify your Sigma System account.
                                            </p>

                                            <h3>Welcome to Sigma System</h3>
                                            <p>Please verify your email by clicking the link below:</p>
                                            <p><a href="{verify_link}">Verify Email Address</a></p>

                                            <p>If you did not create this account, you can safely ignore this email.</p>
                                            """
                                        )
                            
                            if email_sent:
                                st.success("✅ Account created! Please check your email to verify your account.")
                                st.info("Once verified, you can log in with your credentials.")
                            else:
                                st.warning("Account created but email verification failed. Please contact support.")
                        except Exception as e:
                            st.error(f"Error creating account: {str(e)}")
        
        with tab3:
            st.subheader("Forgot Password")
            with st.form("forgot_password_form"):
                reset_email = st.text_input("Enter your email address")
                submit = st.form_submit_button("Send Reset Link")
                
                if submit:
                    with get_db() as conn:
                        row = conn.execute(
                            "SELECT username FROM users WHERE email = ? AND email_verified = 1",
                            (reset_email,)
                        ).fetchone()
                    
                    if row:
                        token = create_token(row[0])
                        app_url = st.secrets.get("APP_URL", "http://localhost:8501")
                        reset_link = f"{app_url}?reset={token}"
                        
                        email_sent = send_email_simple(
                            reset_email,
                            "Reset Your Password - Portfolio Strategy App",
                            f"""
                            <h3>Password Reset Request</h3>
                            <p>You requested to reset your password. Click the link below:</p>
                            <p><a href="{reset_link}">Reset Password</a></p>
                            <p>This link will expire in 30 minutes.</p>
                            <p>If you didn't request this, you can safely ignore this email.</p>
                            """
                        )
                        
                        if email_sent:
                            st.success("Password reset link sent to your email.")
                        else:
                            st.error("Failed to send reset email. Please try again later.")
                    else:
                        st.error("No verified account found with that email.")
        
        st.stop()  # Stop here if not authenticated
    
    # ============================================================
    # AUTHENTICATED USER INTERFACE
    # ============================================================
    
    # Logout button in sidebar
    with st.sidebar:
        st.title(f"Welcome {st.session_state.name}!")
        if st.button("🚪 Logout", type="primary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main app content
    show_strategy_overview()
    st.markdown("---")
    
    # Load user preferences
    if "prefs" not in st.session_state:
        st.session_state.prefs = load_user_prefs(st.session_state.username)
    
    prefs = st.session_state.prefs
    
    # Strategy settings in sidebar
    st.sidebar.header("⚙️ Strategy Settings")
    
    with st.sidebar.expander("Portfolio Configuration", expanded=True):
        start = st.text_input("Start Date", prefs["start_date"])
        risk_on_tickers_str = st.text_input("Risk On Tickers", prefs["risk_on_tickers"])
        risk_on_weights_str = st.text_input("Risk On Weights", prefs["risk_on_weights"])
        risk_off_tickers_str = st.text_input("Risk Off Tickers", prefs["risk_off_tickers"])
        risk_off_weights_str = st.text_input("Risk Off Weights", prefs["risk_off_weights"])
        annual_drag = st.number_input("Annual Drag %", value=float(prefs["annual_drag_pct"]))
    
    with st.sidebar.expander("Portfolio Values", expanded=True):
        qs_cap_1 = st.number_input("Portfolio Value at Last Rebalance", value=float(prefs["qs_cap_1"]))
        real_cap_1 = st.number_input("Portfolio Value Today", value=float(prefs["real_cap_1"]))
    
    with st.sidebar.expander("Advanced Settings", expanded=False):
        inception_date = st.text_input("Inception Date", prefs["official_inception_date"])
        benchmark = st.text_input("Benchmark", prefs["benchmark_ticker"])
        min_days = st.number_input("Confirmation Days", value=int(prefs["min_holding_days"]), min_value=1)
    
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
        st.stop()
    
    # ============================================================
    # TRADING STRATEGY EXECUTION (From your original code)
    # ============================================================
    
    # Process inputs
    try:
        risk_on_tickers = [t.strip().upper() for t in risk_on_tickers_str.split(",")]
        risk_on_weights_list = [float(x) for x in risk_on_weights_str.split(",")]
        risk_on_weights = dict(zip(risk_on_tickers, risk_on_weights_list))
        
        risk_off_tickers = [t.strip().upper() for t in risk_off_tickers_str.split(",")]
        risk_off_weights_list = [float(x) for x in risk_off_weights_str.split(",")]
        risk_off_weights = dict(zip(risk_off_tickers, risk_off_weights_list))
        
        annual_drag_decimal = annual_drag / 100.0
        
        all_tickers = sorted(set(risk_on_tickers + risk_off_tickers))
        prices = load_price_data(all_tickers, start).dropna(how="any")
        
        if len(prices) == 0:
            st.error("No data loaded. Please check your ticker symbols and date range.")
            st.stop()
        
        # Show loading progress
        with st.spinner("Running analysis..."):
            
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
            
            # Run backtests
            best_result = backtest(prices, sig, risk_on_weights, risk_off_weights, FLIP_COST, 
                                  ma_flip_multiplier=3.0, annual_drag_pct=annual_drag_decimal)
            
            latest_signal = sig.iloc[-1]
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
            
            # Calendar quarter logic
            dates = prices.index
            true_q_ends = pd.date_range(start=dates.min(), end=dates.max(), freq='Q')
            mapped_q_ends = []
            for qd in true_q_ends:
                valid_dates = dates[dates <= qd]
                if len(valid_dates) > 0:
                    mapped_q_ends.append(valid_dates.max())
            
            mapped_q_ends = pd.to_datetime(mapped_q_ends)
            
            today_date = pd.Timestamp.today().normalize()
            true_next_q = pd.date_range(start=today_date, periods=2, freq="Q")[0]
            next_q_end = true_next_q
            days_to_next_q = (next_q_end - today_date).days
            
            if len(risk_on_eq) > 0 and risk_on_eq.iloc[0] != 0:
                bh_cagr = (risk_on_eq.iloc[-1] / risk_on_eq.iloc[0]) ** (252 / len(risk_on_eq)) - 1
                quarterly_target = (1 + bh_cagr) ** (1/4) - 1
            else:
                bh_cagr = 0
                quarterly_target = 0
            
            risk_off_daily = pd.Series(0.0, index=simple_rets.index)
            for a, w in risk_off_weights.items():
                if a in simple_rets.columns:
                    risk_off_daily += simple_rets[a] * w
            
            pure_sig_signal = pd.Series(True, index=risk_on_simple.index)
            
            pure_sig_eq, pure_sig_rw, pure_sig_sw, pure_sig_rebals = run_sig_engine(
                risk_on_simple,
                risk_off_daily,
                quarterly_target,
                pure_sig_signal,
                quarter_end_dates=mapped_q_ends,
                quarterly_multiplier=2.0,
                ma_flip_multiplier=0.0
            )
            
            hybrid_eq, hybrid_rw, hybrid_sw, hybrid_rebals = run_sig_engine(
                risk_on_simple,
                risk_off_daily,
                quarterly_target,
                sig,
                pure_sig_rw=pure_sig_rw,
                pure_sig_sw=pure_sig_sw,
                quarter_end_dates=mapped_q_ends,
                quarterly_multiplier=2.0,
                ma_flip_multiplier=3.0
            )
            
            hybrid_simple = hybrid_eq.pct_change().fillna(0)
            
            # Since-inception analysis
            inception = pd.to_datetime(inception_date)
            sigma_eq_si = hybrid_eq.loc[hybrid_eq.index >= inception]
            sigma_ret_si = sigma_eq_si.pct_change().fillna(0)
            
            bh_eq_si = risk_on_eq.loc[risk_on_eq.index >= inception]
            bh_ret_si = bh_eq_si.pct_change().fillna(0)
            
            benchmark_px = load_price_data([benchmark], inception)
            if not benchmark_px.empty and benchmark in benchmark_px.columns:
                benchmark_eq_si = (benchmark_px[benchmark] / benchmark_px[benchmark].iloc[0]).reindex(sigma_eq_si.index).ffill()
                benchmark_ret_si = benchmark_eq_si.pct_change().fillna(0)
            else:
                benchmark_eq_si = pd.Series(1.0, index=sigma_eq_si.index)
                benchmark_ret_si = pd.Series(0.0, index=sigma_eq_si.index)
        
        # ============================================================
        # DISPLAY RESULTS
        # ============================================================
        
        st.success(f"✅ Analysis complete! Data loaded from {prices.index[0].date()} to {prices.index[-1].date()}")
        
        # Current regime
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current 200-Day SMA Regime", current_regime)
        with col2:
            st.metric("Portfolio Value Today", f"${real_cap_1:,.2f}")
        with col3:
            st.metric("Quarterly Target", f"{quarterly_target:.2%}")
        
        # Performance comparison
        st.subheader(f"Performance (Sigma vs Buy & Hold vs {benchmark})")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sigma_eq_si / sigma_eq_si.iloc[0], label="Sigma", linewidth=2, color="blue")
        ax.plot(bh_eq_si / bh_eq_si.iloc[0], label="Buy & Hold", linewidth=2, alpha=0.7)
        ax.plot(benchmark_eq_si, label=benchmark, linewidth=2, linestyle="--", color="black", alpha=0.7)
        ax.set_ylabel("Growth of $1")
        ax.set_title(f"Performance Since {inception_date}")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        # Performance metrics
        sigma_perf_si = compute_enhanced_performance(sigma_ret_si, sigma_eq_si)
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
            ("Sortino", "Sortino", "dec"),
            ("Max Drawdown", "MaxDrawdown", "pct"),
            ("Total Return", "TotalReturn", "pct"),
        ]
        
        for label, key, kind in metrics:
            metrics_data.append([
                label,
                fmt(sigma_perf_si[key], kind),
                fmt(bh_perf_si[key], kind),
                fmt(benchmark_perf_si[key], kind),
            ])
        
        metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Sigma", "Buy & Hold", benchmark])
        st.dataframe(metrics_df, use_container_width=True)
        
        # Rebalance recommendations
        st.subheader("📊 Rebalance Recommendations")
        
        if len(hybrid_rebals) > 0:
            quarter_start_date = hybrid_rebals[-1]
            risky_start = qs_cap_1 * float(hybrid_rw.loc[quarter_start_date])
            risky_today = real_cap_1 * float(hybrid_rw.iloc[-1])
            progress = compute_quarter_progress(risky_start, risky_today, quarterly_target)
            
            gap = progress['Gap ($)']
            date_str = next_q_end.strftime("%m/%d/%Y")
            days_str = f"{days_to_next_q} days"
            dollar_amount = f"${abs(gap):,.2f}"
            
            if gap > 0:
                st.warning(f"**Action Needed:** Sell {dollar_amount} of Risk Off and Buy {dollar_amount} of Risk On")
                st.info(f"**Next Rebalance:** {date_str} ({days_str})")
            elif gap < 0:
                st.warning(f"**Action Needed:** Sell {dollar_amount} of Risk On and Buy {dollar_amount} of Risk Off")
                st.info(f"**Next Rebalance:** {date_str} ({days_str})")
            else:
                st.success(f"**No rebalance needed until {date_str} ({days_str})**")
            
            # Show progress table
            progress_df = pd.DataFrame.from_dict(progress, orient='index', columns=['Value'])
            progress_df.loc["Gap (%)"] = f"{progress['Gap (%)']:.2%}"
            st.dataframe(progress_df)
        
        # Allocation tables
        st.subheader("💼 Portfolio Allocations")
        
        hyb_r = float(hybrid_rw.iloc[-1]) if len(hybrid_rw) > 0 else 0
        hyb_s = float(hybrid_sw.iloc[-1]) if len(hybrid_sw) > 0 else 0
        pure_r = float(pure_sig_rw.iloc[-1]) if len(pure_sig_rw) > 0 else 0
        pure_s = float(pure_sig_sw.iloc[-1]) if len(pure_sig_sw) > 0 else 0
        
        def compute_allocations(account_value, risky_w, safe_w, ron_w, roff_w):
            risky_dollars = account_value * risky_w
            safe_dollars = account_value * safe_w
            alloc = {"Total Risky $": risky_dollars, "Total Safe $": safe_dollars}
            for t, w in ron_w.items():
                alloc[t] = risky_dollars * w
            for t, w in roff_w.items():
                alloc[t] = safe_dollars * w
            return alloc
        
        tab1, tab2, tab3 = st.tabs(["Sigma", "SIG", "200 Day SMA"])
        with tab1:
            alloc = compute_allocations(real_cap_1, hyb_r, hyb_s, risk_on_weights, risk_off_weights)
            alloc_df = pd.DataFrame.from_dict(alloc, orient="index", columns=["$"])
            total = alloc_df["$"].sum()
            alloc_df["% Portfolio"] = (alloc_df["$"] / total * 100).apply(lambda x: f"{x:.2f}%")
            st.dataframe(alloc_df)
        
        with tab2:
            alloc = compute_allocations(real_cap_1, pure_r, pure_s, risk_on_weights, risk_off_weights)
            alloc_df = pd.DataFrame.from_dict(alloc, orient="index", columns=["$"])
            total = alloc_df["$"].sum()
            alloc_df["% Portfolio"] = (alloc_df["$"] / total * 100).apply(lambda x: f"{x:.2f}%")
            st.dataframe(alloc_df)
        
        with tab3:
            if latest_signal:
                alloc = compute_allocations(real_cap_1, 1.0, 0.0, risk_on_weights, {"SHY": 0})
            else:
                alloc = compute_allocations(real_cap_1, 0.0, 1.0, {}, risk_off_weights)
            alloc_df = pd.DataFrame.from_dict(alloc, orient="index", columns=["$"])
            total = alloc_df["$"].sum()
            alloc_df["% Portfolio"] = (alloc_df["$"] / total * 100).apply(lambda x: f"{x:.2f}%")
            st.dataframe(alloc_df)
        
        # MA distance
        st.subheader("📐 200-Day SMA Crossover Distance")
        if len(opt_ma) > 0 and len(portfolio_index) > 0:
            latest_date = opt_ma.dropna().index[-1]
            P = float(portfolio_index.loc[latest_date])
            MA = float(opt_ma.loc[latest_date])
            upper = MA * (1 + tolerance_decimal)
            lower = MA * (1 - tolerance_decimal)
            
            if latest_signal:
                delta = (P - lower) / P
                st.info(f"**Drop Required for Crossover:** {delta:.2%}")
            else:
                delta = (upper - P) / P
                st.info(f"**Gain Required for Crossover:** {delta:.2%}")
        
        # Monte Carlo Analysis
        st.subheader("🎲 Monte Carlo Stress Testing")
        
        total_current_portfolio = real_cap_1
        strategies_mc = {
            "MA Strategy": {"returns": best_result["returns"], "equity": best_result["equity_curve"], "initial_capital": total_current_portfolio},
            "Buy & Hold": {"returns": risk_on_simple, "equity": risk_on_eq, "initial_capital": total_current_portfolio},
            "Sigma": {"returns": hybrid_simple, "equity": hybrid_eq, "initial_capital": total_current_portfolio},
        }
        
        with st.spinner("Running Monte Carlo simulations..."):
            mc_results = {}
            for name, data in strategies_mc.items():
                if len(data["returns"]) > 100:
                    mc_results[name] = monte_carlo_strategy_analysis(
                        data["returns"], data["equity"], n_sim=5000, periods=252, initial_capital=data["initial_capital"]
                    )
            
            if any(v is not None for v in mc_results.values()):
                mc_fig = plot_monte_carlo_results(mc_results, list(strategies_mc.keys()))
                st.pyplot(mc_fig)
                
                # Show key insights
                st.subheader("📈 Key Insights")
                cols = st.columns(3)
                valid_results = [(name, r) for name, r in mc_results.items() if r is not None]
                
                if valid_results:
                    with cols[0]:
                        safest = min(valid_results, key=lambda x: x[1]['cvar_95'])
                        st.metric("Most Conservative", safest[0])
                    with cols[1]:
                        highest_return = max(valid_results, key=lambda x: x[1]['expected_return'])
                        st.metric("Highest Expected Return", highest_return[0])
                    with cols[2]:
                        highest_prob = max(valid_results, key=lambda x: x[1]['prob_positive'])
                        st.metric("Highest Win Probability", f"{highest_prob[1]['prob_positive']:.1%}")
        
        # Strategy diagnostics
        st.subheader("🔍 Strategy Diagnostics")
        diag_fig = plot_diagnostics(hybrid_eq=hybrid_eq, bh_eq=risk_on_eq, hybrid_signal=sig)
        st.pyplot(diag_fig)
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.info("Please check your inputs and try again.")

# ============================================================
# APP ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

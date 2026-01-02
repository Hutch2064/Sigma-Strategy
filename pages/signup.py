# pages/signup.py
import streamlit as st
import streamlit_authenticator as stauth
import yaml
import datetime
from yaml.loader import SafeLoader

st.set_page_config(page_title="Create Account", layout="centered")

st.title("📝 Create New Account")

# Load existing config
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    # Initialize empty config
    config = {
        "credentials": {"usernames": {}},
        "cookie": {
            "name": "portfolio_auth",
            "key": "default_key_" + str(datetime.datetime.now().timestamp()),
            "expiry_days": 30
        },
        "preauthorized": {"emails": []}
    }

# Form for signup
with st.form("signup_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        username = st.text_input("Username", help="Choose a unique username")
        name = st.text_input("Full Name")
    
    with col2:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password", help="At least 6 characters")
    
    confirm_password = st.text_input("Confirm Password", type="password")
    
    # Portfolio defaults section
    st.markdown("---")
    st.subheader("Portfolio Defaults")
    
    col3, col4 = st.columns(2)
    
    with col3:
        default_start = st.text_input("Default Start Date", value="2020-01-01")
        default_risk_on = st.text_input("Default Risk-On Tickers", value="BITU,QQQU,UGL")
        default_risk_off = st.text_input("Default Risk-Off Ticker", value="SHY")
    
    with col4:
        default_drag = st.number_input("Default Annual Drag (%)", value=0.0, step=0.1, format="%.1f")
    
    submitted = st.form_submit_button("Create Account", type="primary")

if submitted:
    # Validation
    if not all([username, name, email, password, confirm_password]):
        st.error("❌ All fields are required")
    elif password != confirm_password:
        st.error("❌ Passwords don't match")
    elif len(password) < 6:
        st.error("❌ Password must be at least 6 characters")
    elif username in config.get('credentials', {}).get('usernames', {}):
        st.error("❌ Username already exists")
    else:
        try:
            # Hash password
            hashed_password = stauth.Hasher([password]).generate()[0]
            
            # Default portfolio preferences
            default_preferences = {
                "start_date": default_start,
                "end_date": "",
                "risk_on_tickers": default_risk_on,
                "risk_on_weights": "0.3333,0.3333,0.3333",
                "risk_off_tickers": default_risk_off,
                "risk_off_weights": "1.0",
                "annual_drag_pct": float(default_drag),
                "qs_cap_1": 75815.26,
                "qs_cap_2": 10074.83,
                "qs_cap_3": 4189.76,
                "real_cap_1": 68832.42,
                "real_cap_2": 9265.91,
                "real_cap_3": 3930.23,
                "last_saved": datetime.datetime.now().isoformat()
            }
            
            # Initialize config structure if empty
            if 'credentials' not in config:
                config['credentials'] = {'usernames': {}}
            if 'cookie' not in config:
                config['cookie'] = {'name': 'portfolio_auth', 'key': 'key', 'expiry_days': 30}
            if 'preauthorized' not in config:
                config['preauthorized'] = {'emails': []}
            
            # Add new user with preferences
            config['credentials']['usernames'][username] = {
                'email': email,
                'name': name,
                'password': hashed_password,
                'preferences': default_preferences
            }
            
            # Add email to preauthorized
            if email not in config['preauthorized']['emails']:
                config['preauthorized']['emails'].append(email)
            
            # Save config
            with open('config.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            
            st.success("✅ Account created successfully!")
            st.balloons()
            
            # Show credentials
            st.info(f"""
            **Your login credentials:**
            - **Username:** `{username}`
            - **Password:** `{password}`
            
            **Default portfolio settings saved.**
            You can now login and customize your portfolio.
            """)
            
            # Auto-redirect option
            if st.button("🚀 Go to Login Page", type="primary"):
                st.switch_page("app.py")
                
        except Exception as e:
            st.error(f"❌ Error creating account: {str(e)}")

# Link back to login
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("← Back to Login", use_container_width=True):
        st.switch_page("app.py")
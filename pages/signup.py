# pages/signup.py
import streamlit as st
import streamlit_authenticator as stauth
import yaml
import datetime
from yaml.loader import SafeLoader

st.set_page_config(page_title="Create Account")

st.title("📝 Create New Account")

# Load existing config
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    config = {
        "credentials": {"usernames": {}},
        "cookie": {"name": "portfolio_auth", "key": "default_key", "expiry_days": 30},
        "preauthorized": {"emails": []}
    }

# Simple signup form
with st.form("signup_form"):
    username = st.text_input("Username")
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    submitted = st.form_submit_button("Create Account", type="primary")

if submitted:
    if not all([username, name, email, password, confirm_password]):
        st.error("All fields are required")
    elif password != confirm_password:
        st.error("Passwords don't match")
    elif len(password) < 6:
        st.error("Password must be at least 6 characters")
    elif username in config.get('credentials', {}).get('usernames', {}):
        st.error("Username already exists")
    else:
        # Hash password
        hashed_password = stauth.Hasher([password]).generate()[0]
        
        # DEFAULT portfolio settings (simple defaults) - UPDATED for single portfolio
        default_preferences = {
            "start_date": "2020-01-01",
            "end_date": "",
            "risk_on_tickers": "BITU,QQQU,UGL",
            "risk_on_weights": "0.3333,0.3333,0.3333",
            "risk_off_tickers": "SHY",
            "risk_off_weights": "1.0",
            "annual_drag_pct": 0.0,
            "qs_cap_1": 75815.26,
            "real_cap_1": 68832.42,
            "last_saved": datetime.datetime.now().isoformat()
        }
        
        # Add new user with default preferences
        if 'credentials' not in config:
            config['credentials'] = {'usernames': {}}
        
        config['credentials']['usernames'][username] = {
            'email': email,
            'name': name,
            'password': hashed_password,
            'preferences': default_preferences
        }
        
        # Save config
        with open('config.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        st.success("✅ Account created!")
        st.info(f"Username: {username}")
        
        if st.button("Go to Login"):
            st.switch_page("app.py")

# Back to login
st.markdown("---")
if st.button("← Back to Login"):
    st.switch_page("app.py")
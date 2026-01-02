# pages/signup.py
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(page_title="Create Account", layout="centered")

# Center the form
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.title("Create Account")
    
    # Load config
    try:
        with open('config.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)
    except:
        config = {"credentials": {"usernames": {}}, "cookie": {}, "preauthorized": {"emails": []}}
    
    # Form using st.form (correct way)
    with st.form("signup_form"):
        username = st.text_input("Username")
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        # Submit button INSIDE the form
        submit_button = st.form_submit_button("Create Account", type="primary")
        
        if submit_button:
            if not all([username, name, email, password, confirm_password]):
                st.error("All fields are required")
            elif password != confirm_password:
                st.error("Passwords don't match")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters")
            elif username in config.get('credentials', {}).get('usernames', {}):
                st.error("Username already exists")
            else:
                # Create user
                hashed_password = stauth.Hasher([password]).generate()[0]
                
                # Initialize config if empty
                if 'credentials' not in config:
                    config['credentials'] = {'usernames': {}}
                if 'cookie' not in config:
                    config['cookie'] = {'name': 'portfolio_auth', 'key': 'key', 'expiry_days': 30}
                if 'preauthorized' not in config:
                    config['preauthorized'] = {'emails': []}
                
                # Add user
                config['credentials']['usernames'][username] = {
                    'email': email,
                    'name': name,
                    'password': hashed_password
                }
                
                # Save
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
                
                st.success("✅ Account created!")
                st.info(f"Username: {username}")
                
                # Auto-redirect
                import time
                time.sleep(2)
                st.switch_page("app.py")
    
    # Button OUTSIDE the form
    st.markdown("---")
    if st.button("Already have an account? Login"):
        st.switch_page("app.py")
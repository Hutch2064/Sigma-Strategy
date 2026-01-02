# pages/signup.py
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.title("Create Account")

with st.form("signup_form"):
    username = st.text_input("Username")
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    submitted = st.form_submit_button("Create Account")
    
    if submitted:
        if password != confirm_password:
            st.error("Passwords don't match")
        elif len(password) < 6:
            st.error("Password must be at least 6 characters")
        else:
            # Load existing config
            with open('config.yaml') as file:
                config = yaml.load(file, Loader=SafeLoader)
            
            # Check if username exists
            if username in config['credentials']['usernames']:
                st.error("Username already exists")
            else:
                # Hash password
                hashed_password = stauth.Hasher([password]).generate()[0]
                
                # Add new user
                config['credentials']['usernames'][username] = {
                    'email': email,
                    'name': name,
                    'password': hashed_password
                }
                
                # Save back to YAML
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
                
                st.success("Account created! Please login.")
                if st.button("Go to Login"):
                    st.switch_page("app.py")

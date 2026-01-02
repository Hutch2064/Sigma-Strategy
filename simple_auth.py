import streamlit as st
import bcrypt
import json
import os
from datetime import datetime

# Simple file-based user storage (we'll upgrade to database later)
USERS_FILE = "users.json"

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    """Hash a password for storing"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password.encode('utf-8'))

def create_account(email, password, full_name):
    """Create a new user account"""
    users = load_users()
    
    if email in users:
        return False, "Email already registered"
    
    users[email] = {
        "password_hash": hash_password(password),
        "full_name": full_name,
        "created_at": datetime.now().isoformat(),
        "subscription": "free",
        "email_notifications": True
    }
    
    save_users(users)
    return True, "Account created successfully!"

def login(email, password):
    """Authenticate a user"""
    users = load_users()
    
    if email not in users:
        return False, "User not found", None
    
    user = users[email]
    
    if not verify_password(user["password_hash"], password):
        return False, "Invalid password", None
    
    return True, "Login successful", user

def show_auth_page():
    """Show authentication page (login/signup)"""
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    
    # If already authenticated, return user
    if st.session_state.authenticated:
        return st.session_state.user
    
    # Show login or signup form
    if st.session_state.show_signup:
        # SIGNUP FORM
        st.title("Create Account")
        
        with st.form("signup_form"):
            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            agree_tos = st.checkbox("I agree to the Terms of Service")
            
            submit = st.form_submit_button("Create Account")
            
            if submit:
                if not all([full_name, email, password, confirm_password]):
                    st.error("All fields are required")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif not agree_tos:
                    st.error("You must agree to the Terms of Service")
                else:
                    success, message = create_account(email, password, full_name)
                    if success:
                        st.success(message)
                        st.session_state.show_signup = False
                        st.rerun()
                    else:
                        st.error(message)
        
        # Back to login link
        if st.button("Already have an account? Login"):
            st.session_state.show_signup = False
            st.rerun()
    
    else:
        # LOGIN FORM
        st.title("Login to SigmaTrader")
        
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            remember_me = st.checkbox("Remember me")
            
            submit = st.form_submit_button("Login")
            
            if submit:
                if not email or not password:
                    st.error("Email and password are required")
                else:
                    success, message, user = login(email, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.session_state.user['email'] = email  # Store email in user dict
                        st.success(f"Welcome back, {user['full_name']}!")
                        st.rerun()
                    else:
                        st.error(message)
        
        # Signup link
        if st.button("Don't have an account? Sign up"):
            st.session_state.show_signup = True
            st.rerun()
    
    # Show stop if not authenticated
    st.stop()

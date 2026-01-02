# preferences_manager.py
import yaml
import datetime
from yaml.loader import SafeLoader

def save_user_preferences(username, preferences):
    """Save user's current portfolio inputs"""
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.load(file, Loader=SafeLoader)
        
        # Add timestamp
        preferences['last_saved'] = datetime.datetime.now().isoformat()
        
        # Update user's preferences
        if username in config['credentials']['usernames']:
            config['credentials']['usernames'][username]['preferences'] = preferences
            
            # Save back
            with open('config.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            return True
    except Exception as e:
        print(f"Error saving preferences: {e}")
    return False

def load_user_preferences(username):
    """Load user's saved portfolio inputs"""
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.load(file, Loader=SafeLoader)
        
        if username in config['credentials']['usernames']:
            return config['credentials']['usernames'][username].get('preferences', {})
    except:
        pass
    return {}

def get_default_preferences():
    """Get default portfolio settings"""
    return {
        "start_date": "2020-01-01",
        "end_date": "",
        "risk_on_tickers": "TQQQ",
        "risk_on_weights": "1.0",
        "risk_off_tickers": "AGG",
        "risk_off_weights": "1.0",
        "annual_drag_pct": 0.0,
        "qs_cap_1": 10000,
        "real_cap_1": 10000
        "official_inception_date": "2025-12-22",
        "benchmark_ticker": "QQQ",
    }
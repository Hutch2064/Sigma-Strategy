# create_default_admin.py
"""
One-time script to create admin user if config.yaml doesn't exist
Run this once: python create_default_admin.py
"""
import streamlit_authenticator as stauth
import yaml
import os

print("=== Creating Default Admin User ===")

# Hash the admin password
hashed_password = stauth.Hasher(["admin123"]).generate()[0]

# Create config structure
config = {
    "credentials": {
        "usernames": {
            "admin": {
                "email": "admin@example.com",
                "name": "Administrator",
                "password": hashed_password,
                "preferences": {
                    "start_date": "2020-01-01",
                    "end_date": "",
                    "risk_on_tickers": "BITU,QQQU,UGL",
                    "risk_on_weights": "0.3333,0.3333,0.3333",
                    "risk_off_tickers": "SHY",
                    "risk_off_weights": "1.0",
                    "annual_drag_pct": 0.0,
                    "qs_cap_1": 75815.26,
                    "qs_cap_2": 10074.83,
                    "qs_cap_3": 4189.76,
                    "real_cap_1": 68832.42,
                    "real_cap_2": 9265.91,
                    "real_cap_3": 3930.23,
                    "last_saved": "2024-01-01T00:00:00"
                }
            }
        }
    },
    "cookie": {
        "name": "portfolio_auth",
        "key": "change_this_to_random_key_in_production",
        "expiry_days": 30
    },
    "preauthorized": {
        "emails": ["admin@example.com"]
    }
}

# Save to YAML
with open('config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

print("✅ config.yaml created successfully!")
print("\n🔑 Default Admin Credentials:")
print("   Username: admin")
print("   Password: admin123")
print("\n⚠️  IMPORTANT:")
print("   1. Change the cookie 'key' in config.yaml for production")
print("   2. Change the admin password after first login")
print("   3. Never commit real passwords to GitHub")

if os.path.exists('config.yaml'):
    print(f"\n📁 File created: {os.path.abspath('config.yaml')}")

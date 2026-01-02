# create_users.py
import streamlit_authenticator as stauth
import yaml

# Define your users (username, name, plain_password)
users = [
    ("admin", "Admin User", "admin123"),
    ("user1", "Test User", "password123"),
]

# Hash passwords
hashed_passwords = stauth.Hasher(["admin123", "password123"]).generate()

# Create config dictionary
config = {
    "credentials": {
        "usernames": {
            "admin": {
                "email": "admin@example.com",
                "name": "Admin User",
                "password": hashed_passwords[0]
            },
            "user1": {
                "email": "user1@example.com",
                "name": "Test User",
                "password": hashed_passwords[1]
            }
        }
    },
    "cookie": {
        "name": "portfolio_auth",
        "key": "some_random_key_123",
        "expiry_days": 30
    },
    "preauthorized": {
        "emails": ["admin@example.com"]
    }
}

# Save to YAML file
with open('config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

print("✅ config.yaml created successfully!")
print("Usernames and passwords:")
print("  - admin / admin123")
print("  - user1 / password123")

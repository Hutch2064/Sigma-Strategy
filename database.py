# database.py
from deta import Deta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Deta project key from environment variable
DETA_KEY = os.getenv("DETA_KEY")

if not DETA_KEY:
    raise ValueError("DETA_KEY not found. Please add it to your .env file")

# Initialize Deta
deta = Deta(DETA_KEY)

# Create/connect to database
db = deta.Base("users_db")

def insert_user(username, name, password):
    return db.put({
        "key": username,
        "name": name,
        "password": password
    })

def fetch_all_users():
    res = db.fetch()
    return res.items

def get_user(username):
    return db.get(username)

def update_user(username, updates):
    return db.update(updates, username)

def delete_user(username):
    return db.delete(username)

# preferences_manager.py
import streamlit as st
import yaml
import datetime
from yaml.loader import SafeLoader

class PortfolioPreferences:
    """Manage user portfolio preferences storage and retrieval"""
    
    @staticmethod
    def get_default_preferences():
        """Return default portfolio preferences"""
        return {
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
            "last_saved": datetime.datetime.now().isoformat()
        }
    
    @staticmethod
    def get_preference_keys():
        """Return list of all preference keys"""
        return list(PortfolioPreferences.get_default_preferences().keys())
    
    @staticmethod
    def save_preferences(username, preferences_dict):
        """
        Save user preferences to config.yaml
        
        Args:
            username: Username to save preferences for
            preferences_dict: Dictionary of preferences to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load current config
            with open('config.yaml', 'r') as file:
                config = yaml.load(file, Loader=SafeLoader)
            
            # Check if user exists
            if username not in config.get('credentials', {}).get('usernames', {}):
                st.error(f"User '{username}' not found in config")
                return False
            
            # Add timestamp
            preferences_dict['last_saved'] = datetime.datetime.now().isoformat()
            
            # Update user preferences
            config['credentials']['usernames'][username]['preferences'] = preferences_dict
            
            # Save back to file
            with open('config.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            
            return True
            
        except FileNotFoundError:
            st.error("Config file not found. Please restart the app.")
            return False
        except Exception as e:
            st.error(f"Error saving preferences: {str(e)}")
            return False
    
    @staticmethod
    def load_preferences(username):
        """
        Load user preferences from config.yaml
        
        Args:
            username: Username to load preferences for
            
        Returns:
            dict: User preferences or empty dict if not found
        """
        try:
            with open('config.yaml', 'r') as file:
                config = yaml.load(file, Loader=SafeLoader)
            
            # Get user preferences if they exist
            user_data = config.get('credentials', {}).get('usernames', {}).get(username, {})
            preferences = user_data.get('preferences', {})
            
            # Merge with defaults to ensure all keys exist
            defaults = PortfolioPreferences.get_default_preferences()
            merged_preferences = {**defaults, **preferences}
            
            return merged_preferences
            
        except FileNotFoundError:
            st.error("Config file not found.")
            return PortfolioPreferences.get_default_preferences()
        except Exception as e:
            st.error(f"Error loading preferences: {str(e)}")
            return PortfolioPreferences.get_default_preferences()
    
    @staticmethod
    def save_current_preferences(username, st_session_state):
        """
        Save current session state as user preferences
        
        Args:
            username: Username to save for
            st_session_state: Streamlit session state object
            
        Returns:
            bool: True if successful
        """
        try:
            preferences = {}
            for key in PortfolioPreferences.get_preference_keys():
                # Try to get from session state, fall back to default
                value = st_session_state.get(key, PortfolioPreferences.get_default_preferences().get(key))
                preferences[key] = value
            
            return PortfolioPreferences.save_preferences(username, preferences)
            
        except Exception as e:
            st.error(f"Error collecting preferences: {str(e)}")
            return False
    
    @staticmethod
    def load_to_session_state(username, st_session_state):
        """
        Load user preferences into session state
        
        Args:
            username: Username to load for
            st_session_state: Streamlit session state object
            
        Returns:
            dict: Loaded preferences
        """
        preferences = PortfolioPreferences.load_preferences(username)
        
        # Update session state
        for key, value in preferences.items():
            st_session_state[key] = value
        
        return preferences
    
    @staticmethod
    def initialize_session_state(st_session_state, username=None):
        """
        Initialize session state with preferences
        
        Args:
            st_session_state: Streamlit session state
            username: Optional username to load preferences from
        """
        # Initialize all preference keys in session state
        for key in PortfolioPreferences.get_preference_keys():
            if key not in st_session_state:
                # Load from user if provided, otherwise use defaults
                if username:
                    prefs = PortfolioPreferences.load_preferences(username)
                    st_session_state[key] = prefs.get(key, PortfolioPreferences.get_default_preferences()[key])
                else:
                    st_session_state[key] = PortfolioPreferences.get_default_preferences()[key]

import requests
import json
import urllib3
import logging
from urllib.parse import urlencode
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Disable SSL warnings
urllib3.disable_warnings()

base_url = "https://soccerapi.pkbipcas.com"

def check_server_status():
    """Check if the server is responding"""
    try:
        response = requests.get(f"{base_url}/health", verify=False)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Server health check failed: {e}")
        return False

def test_login_request(test_name, payload, content_type=None):
    """Helper function to test login requests with different formats"""
    print(f"Test: {test_name}")

    headers = {
        'Content-Type': content_type,
        'Accept': 'application/json'
    } if content_type else {'Accept': 'application/json'}

    try:
        if content_type == "application/x-www-form-urlencoded":
            # data = urlencode(payload) # This line is removed or commented out
            response = requests.post(
                f"{base_url}/auth/login",
                data=payload,  # Pass the dictionary directly
                headers=headers,
                verify=False
            )
        else:
            response = requests.post(
                f"{base_url}/auth/login",
                json=payload,  # Let requests handle JSON serialization
                headers=headers,
                verify=False
            )

        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
        print(f"Request Headers: {headers}")
        print(f"Request Payload: {payload}")

        if test_name == "Form data request":
            if response.status_code == 422:
                logger.info("Form data request correctly received a 422 error (as expected, endpoint is JSON-only).")
            else:
                logger.error(f"Form data request received status {response.status_code}, but expected 422.")
                logger.debug(f"Status Code: {response.status_code}")
                logger.debug(f"Response: {response.text}")
                logger.debug(f"Request payload: {payload}")
                logger.debug(f"Request headers: {headers}")
        elif response.status_code >= 400: # For JSON request or other unexpected errors
            logger.error(f"Error occurred during {test_name}")
            logger.debug(f"Status Code: {response.status_code}")
            logger.debug(f"Response: {response.text}")
            logger.debug(f"Request payload: {payload}")
            logger.debug(f"Request headers: {headers}")

        return response

    except Exception as e:
        logger.error(f"Request failed during {test_name}: {str(e)}")
        return None

def main():
    # Check server status first
    if not check_server_status():
        logger.error("Server is not responding. Please check if it's running.")
        return

    print("Testing different login request formats...")

    credentials = {
        "username_or_email": "test_user5",
        "password": "secure_password_123"
    }

    timestamp = int(time.time())
    test_username = f"test_user_{timestamp}"
    test_email = f"test_user_{timestamp}@example.com"
    test_password = "a_very_secure_password_123!"

    # Test 1: Register a new user
    print(f"Test: Registering new user: {test_username}")
    register_payload = {
        "username": test_username,
        "email": test_email,
        "password": test_password
    }
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    try:
        response = requests.post(
            f"{base_url}/auth/register",
            json=register_payload,
            headers=headers,
            verify=False # In production, use verify=True with proper certs
        )
        print(f"Register Status: {response.status_code}")
        print(f"Register Response: {response.text[:200]}")

        if response.status_code == 200 or response.status_code == 201: # Assuming 200 or 201 for successful registration
            print("Registration successful or user might have been created just now.")

            # Test 2: Login with the newly registered user
            print(f"Test: Logging in with new user: {test_username}")
            login_payload = {
                "username_or_email": test_username, # or test_email
                "password": test_password
            }
            login_response = requests.post(
                f"{base_url}/auth/login",
                json=login_payload,
                headers=headers,
                verify=False
            )
            print(f"Login Status: {login_response.status_code}")
            print(f"Login Response: {login_response.text[:200]}")

            if login_response.status_code == 200:
                logger.info(f"Successfully logged in as {test_username}!")
            else:
                logger.error(f"Failed to log in as {test_username} after registration.")
                logger.debug(f"Login Status: {login_response.status_code}, Response: {login_response.text}")

        else:
            logger.error(f"Registration failed for {test_username}. Status: {response.status_code}, Response: {response.text}")

    except Exception as e:
        logger.error(f"Request failed during registration/login test: {str(e)}")

    # ... (You can keep your form data test if you want to ensure it still returns 422)
    print("\nTest: Form data request (expected 422)")
    form_credentials = {
        "username_or_email": test_username, # using the same username
        "password": test_password
    }
    test_login_request( # Your existing helper function
        "Form data request",
        form_credentials,
        "application/x-www-form-urlencoded"
    )

if __name__ == "__main__":
    main()
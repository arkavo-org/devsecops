import requests
import env
import jwt

# Keycloak configuration
KEYCLOAK_REALM = "master"  # Replace with your realm name
KEYCLOAK_URL = f"{env.KEYCLOAK_AUTH_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"
CLIENT_ID = "admin-cli"  # Admin CLI is typically used for admin accounts
USERNAME = "admin"  # Replace with the Keycloak admin username
PASSWORD = "changeme"  # Replace with the Keycloak admin password

# Obtain the token
def get_token():
    data = {
        "grant_type": "password",
        "client_id": CLIENT_ID,
        "username": USERNAME,
        "password": PASSWORD,
    }

    try:
        response = requests.post(KEYCLOAK_URL, data=data)
        response.raise_for_status()
        token_data = response.json()
        return token_data.get("access_token")
    except requests.exceptions.RequestException as e:
        print(f"Error obtaining token: {e}")
        return None

# Test the token with an API endpoint
def test_token(token):
    test_url = env.PROTOCOL_SYNAPSE_BASE_URL + "/_matrix/client/r0/joined_rooms"  # Replace with your Matrix API URL
    test_url = env.PROTOCOL_SYNAPSE_BASE_URL + "/_matrix/client/versions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(test_url, headers=headers)
        response.raise_for_status()
        print("API Response:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error testing token: {e}")
        if response.status_code == 401:
            print("Token may be invalid or expired.")

# Main script
if __name__ == "__main__":
    print("Obtaining Bearer Token for Admin...")
    access_token = get_token()

    if access_token:
        print("Admin token obtained successfully!")
        decoded = jwt.decode(access_token, options={"verify_signature": False})
        print(decoded)
        print("Testing the token with Matrix API...")
        test_token(access_token)
    else:
        print("Failed to obtain admin token.")

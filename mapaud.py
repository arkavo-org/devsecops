import requests
import env

# Keycloak server configuration
KEYCLOAK_URL = "https://keycloak.app.codecollective.us/auth"  # Your Keycloak server URL
REALM_NAME = "opentdf"  # The realm name
ADMIN_USER = "admin"  # Keycloak admin username
ADMIN_PASS = "changeme"  # Keycloak admin password
CLIENT_ID = "web-client"  # The client ID for which to adjust the audience
SYNAPSE_CLIENT_ID = "synapse"  # The Synapse client ID
AUDIENCE_MAPPER_NAME = "Synapse Audience Mapper"


# Authenticate with Keycloak to get an admin access token
def get_admin_token():
    auth_url = f"{KEYCLOAK_URL}/realms/master/protocol/openid-connect/token"
    data = {
        "client_id": "admin-cli",
        "username": ADMIN_USER,
        "password": ADMIN_PASS,
        "grant_type": "password",
    }
    response = requests.post(auth_url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

# Create an audience mapper for the client
def create_audience_mapper(admin_token, mapper_url):
    headers = {"Authorization": f"Bearer {admin_token}", "Content-Type": "application/json"}
    
    # Define the audience mapper
    audience_mapper = {
        "name": AUDIENCE_MAPPER_NAME,
        "protocol": "openid-connect",
        "protocolMapper": "oidc-audience-mapper",
        "consentRequired": False,
        "config": {
            "included.client.audience": SYNAPSE_CLIENT_ID,
            "id.token.claim": "true",
            "access.token.claim": "true",
        }
    }
    
    response = requests.post(mapper_url, headers=headers, json=audience_mapper)
    response.raise_for_status()
    print("Audience mapper created successfully.")

def get_client_uuid(admin_token, client_id):
    headers = {"Authorization": f"Bearer {admin_token}"}
    clients_url = f"{KEYCLOAK_URL}/admin/realms/{REALM_NAME}/clients"
    response = requests.get(clients_url, headers=headers)
    response.raise_for_status()
    clients = response.json()
    for client in clients:
        if client["clientId"] == client_id:
            return client["id"]
    raise ValueError(f"Client ID {client_id} not found in realm {REALM_NAME}")

def main():
    admin_token = get_admin_token()
    client_uuid = get_client_uuid(admin_token, CLIENT_ID)
    mapper_url = f"{KEYCLOAK_URL}/admin/realms/{REALM_NAME}/clients/{client_uuid}/protocol-mappers/models"

    create_audience_mapper(admin_token, mapper_url)

if __name__ == "__main__":
    main()

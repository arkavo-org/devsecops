#!/bin/bash

# Check if required environment variables are set
if [[ -z "$KEYCLOAK_ADMIN" || -z "$KEYCLOAK_ADMIN_PASSWORD" || -z "$KEYCLOAK_SERVER_URL" ]]; then
  echo "Please ensure KEYCLOAK_ADMIN, KEYCLOAK_ADMIN_PASSWORD, and KEYCLOAK_SERVER_URL environment variables are set."
  exit 1
fi

# Variables
REALM="opentdf"
NEW_CLIENT_ID="arkavo-admin"
KEYCLOAK_REDIRECT_URI="${KEYCLOAK_REDIRECT_URI:-https://localhost:8443/auth/*}" # Default redirect URI if not set
KEYCLOAK_REDIRECT_URI="https://localhost:8443/auth/*"

# Get admin access token
ACCESS_TOKEN=$(curl -s \
  -d "client_id=admin-cli" \
  -d "username=$KEYCLOAK_ADMIN" \
  -d "password=$KEYCLOAK_ADMIN_PASSWORD" \
  -d "grant_type=password" \
  "$KEYCLOAK_SERVER_URL/auth/realms/master/protocol/openid-connect/token" | jq -r '.access_token')

if [[ -z "$ACCESS_TOKEN" || "$ACCESS_TOKEN" == "null" ]]; then
  echo "Failed to obtain access token. Check your credentials."
  exit 1
fi

echo "Access token retrieved."

echo "Connecting to $KEYCLOAK_SERVER_URL/admin/realms/$REALM/clients"
# Create the client
CLIENT_RESPONSE=$(curl -s -X POST \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "clientId": "'"$NEW_CLIENT_ID"'",
    "enabled": true,
    "directAccessGrantsEnabled": true,
    "publicClient": false,
    "serviceAccountsEnabled": true,
    "standardFlowEnabled": true,
    "redirectUris": ["'"$KEYCLOAK_REDIRECT_URI"'"]
  }' \
  "$KEYCLOAK_SERVER_URL/admin/realms/$REALM/clients")

echo "Client creation response: $CLIENT_RESPONSE"

# Extract the client UUID
CLIENT_UUID=$(echo "$CLIENT_RESPONSE" | jq -r '.id')

if [[ -z "$CLIENT_UUID" || "$CLIENT_UUID" == "null" ]]; then
  echo "Failed to create client or retrieve its ID."
  exit 1
fi

echo "Client '$NEW_CLIENT_ID' created with ID: $CLIENT_UUID"

# Get client secret
CLIENT_SECRET=$(curl -s -X GET \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  "$KEYCLOAK_SERVER_URL/auth/realms/$REALM/clients/$CLIENT_UUID/client-secret" | jq -r '.value')

if [[ -z "$CLIENT_SECRET" || "$CLIENT_SECRET" == "null" ]]; then
  echo "Failed to retrieve client secret."
  exit 1
fi

echo "Client secret: $CLIENT_SECRET"

# Assign roles to the service account
SERVICE_ACCOUNT_USER_ID=$(curl -s -X GET \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  "$KEYCLOAK_SERVER_URL/auth/realms/$REALM/clients/$CLIENT_UUID/service-account-user" | jq -r '.id')

if [[ -z "$SERVICE_ACCOUNT_USER_ID" || "$SERVICE_ACCOUNT_USER_ID" == "null" ]]; then
  echo "Failed to retrieve service account user ID."
  exit 1
fi

echo "Service account user ID: $SERVICE_ACCOUNT_USER_ID"

# Assign the necessary roles
curl -s -X POST \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '[
    {"id": "admin", "name": "admin"},
    {"id": "manage-users", "name": "manage-users"},
    {"id": "create-realm", "name": "create-realm"}
  ]' \
  "$KEYCLOAK_SERVER_URL/auth/realms/$REALM/users/$SERVICE_ACCOUNT_USER_ID/role-mappings/realm"

echo "Roles assigned to service account."

# Add custom client scopes (if needed)
curl -s -X POST \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "custom-backend-scope",
    "protocol": "openid-connect",
    "attributes": {
      "include.in.token.scope": "true",
      "display.on.consent.screen": "false"
    }
  }' \
  "$KEYCLOAK_SERVER_URL/auth/realms/$REALM/client-scopes"

echo "Custom client scope added."

# Final Output
echo "Client '$NEW_CLIENT_ID' successfully configured with backend permissions."
echo "Client ID: $CLIENT_UUID"
echo "Client Secret: $CLIENT_SECRET"

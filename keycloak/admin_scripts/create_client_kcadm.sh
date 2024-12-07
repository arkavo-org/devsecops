#!/bin/bash

# Authenticate with Keycloak
kcadm config credentials \
  --server "$KEYCLOAK_SERVER_URL/auth" \
  --realm master \
  --user "$KEYCLOAK_ADMIN" \
  --password "$KEYCLOAK_ADMIN_PASSWORD"

if [ $? -ne 0 ]; then
  echo "Failed to authenticate with Keycloak."
  exit 1
fi

echo "Authenticated with Keycloak."

# Fetch all roles in the realm
ALL_ROLES=$(kcadm get roles -r "$REALM" --fields 'name,composite')
export ALL_ROLES

# Check if 'broad-admin' role exists
BROAD_ADMIN_ROLE=$(echo "$ALL_ROLES" | jq -r '.[] | select(.name == "broad-admin") | .name')

if [ -z "$BROAD_ADMIN_ROLE" ]; then
  echo "Broad admin role 'broad-admin' does not exist. Creating it now..."

  # Create the 'broad-admin' role
  kcadm create roles -r "$REALM" \
    -s name="broad-admin" \
    -s description="Broad admin access role with permissions to manage the realm."

  if [ $? -ne 0 ]; then
    echo "Failed to create 'broad-admin' role."
    exit 1
  fi

  echo "'broad-admin' role created."

  # Add 'realm-admin' as a composite role to 'broad-admin'
  echo "Adding 'realm-admin' as a composite role to 'broad-admin'..."
  REALM_ADMIN_ROLE_ID=$(kcadm get roles -r "$REALM" --rolename "realm-admin" --fields id | jq -r '.id')

  if [ -z "$REALM_ADMIN_ROLE_ID" ]; then
    echo "Failed to find 'realm-admin' role ID."
    exit 1
  fi

  kcadm update roles -r "$REALM" "broad-admin" \
    -s "composites.realm=[\"$REALM_ADMIN_ROLE_ID\"]"

  if [ $? -ne 0 ]; then
    echo "Failed to add 'realm-admin' as a composite role to 'broad-admin'."
    exit 1
  fi

  echo "'broad-admin' configured with 'realm-admin' privileges."
else
  echo "Broad admin role 'broad-admin' already exists."
fi

# Fetch all clients and export to a variable
ALL_CLIENTS=$(kcadm get clients -r "$REALM" --fields 'id,clientId')
export ALL_CLIENTS

# Extract the UUID of the existing client if it exists
EXISTING_CLIENT_UUID=$(echo "$ALL_CLIENTS" | jq -r ".[] | select(.clientId == \"$CLIENT_ID\") | .id")

if [ -n "$EXISTING_CLIENT_UUID" ]; then
  echo "Client '$CLIENT_ID' already exists with ID: $EXISTING_CLIENT_UUID"
  CLIENT_UUID=$EXISTING_CLIENT_UUID
else
  # Create the client
  CLIENT_UUID=$(kcadm create clients -r "$REALM" \
    -s clientId="$CLIENT_ID" \
    -s enabled=true \
    -s publicClient=false \
    -s serviceAccountsEnabled=true \
    -s redirectUris="[\"$REDIRECT_URI\"]" \
    -i)

  if [ -z "$CLIENT_UUID" ]; then
    echo "Failed to create client."
    exit 1
  fi

  echo "Client '$CLIENT_ID' created with ID: $CLIENT_UUID"
fi

# Fetch service account user ID
SERVICE_ACCOUNT_USER_ID=$(kcadm get users -r "$REALM" \
  -q username="service-account-$CLIENT_ID" \
  --fields id \
  | jq -r '.[].id')

if [ -z "$SERVICE_ACCOUNT_USER_ID" ]; then
  echo "Failed to retrieve service account user ID."
  exit 1
fi

echo "Service account user ID: $SERVICE_ACCOUNT_USER_ID"

# Assign the 'broad-admin' role to the service account
kcadm add-roles -r "$REALM" \
  --uusername "service-account-$CLIENT_ID" \
  --rolename "broad-admin"

if [ $? -ne 0 ]; then
  echo "Failed to assign 'broad-admin' role to the service account."
  exit 1
fi

echo "Assigned 'broad-admin' role to the service account."

echo "Service account for '$CLIENT_ID' successfully configured."

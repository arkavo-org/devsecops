#!/bin/bash

# Paths
KEYCLOAK_IMPORT_DIR="/opt/keycloak/data/import"
CERTS_DIR="/opt/keycloak/oauth_certs"
CUSTOM_KEYSTORE="/opt/keycloak/cacerts"  # Use a writable copy of the keystore
KEYSTORE_PASS="changeit"

# Copy the system-wide cacerts to a writable location (if not already copied)
if [ ! -f "$CUSTOM_KEYSTORE" ]; then
    echo "Copying system cacerts to $CUSTOM_KEYSTORE..."
    cp /usr/lib/jvm/java-17-openjdk/lib/security/cacerts "$CUSTOM_KEYSTORE"
    chmod 644 "$CUSTOM_KEYSTORE"
fi

# Import OAuth Certificates into the custom KeyStore
echo "Importing OAuth Certificates..."
for cert in $CERTS_DIR/*.crt; do
    if [ -f "$cert" ]; then
        alias=$(basename "$cert" .crt)
        keytool -import -trustcacerts \
            -keystore "$CUSTOM_KEYSTORE" \
            -storepass "$KEYSTORE_PASS" \
            -noprompt \
            -alias "$alias" \
            -file "$cert"
        echo "Imported $cert as alias $alias"
    fi
done

# Ensure the import directory exists and copy realm files
mkdir -p "$KEYCLOAK_IMPORT_DIR"
echo "Copying realm JSON files to import directory..."
cp /opt/keycloak/realms/*.json "$KEYCLOAK_IMPORT_DIR/"

# Start Keycloak with import-realm flag
echo "Starting Keycloak with import-realm flag..."
exec /opt/keycloak/bin/kc.sh start --import-realm --verbose

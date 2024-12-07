#!/bin/bash

# Variables
KC_VERSION=24.0.5
INSTALL_DIR="$HOME/keycloak-${KC_VERSION}"
LINK_TARGET="/usr/local/bin/kcadm"

# Step 1: Install Keycloak
install_keycloak() {
    echo "Installing Keycloak version ${KC_VERSION}..."
    curl -LO https://github.com/keycloak/keycloak/releases/download/${KC_VERSION}/keycloak-${KC_VERSION}.zip
    unzip -q keycloak-${KC_VERSION}.zip -d "$HOME"
    echo "Keycloak installed at ${INSTALL_DIR}"
}

# Step 2: Link kcadm.sh to a directory in PATH
setup_kcadm_link() {
    echo "Setting up kcadm.sh reference..."
    if [[ -f "${INSTALL_DIR}/bin/kcadm.sh" ]]; then
        sudo ln -sf "${INSTALL_DIR}/bin/kcadm.sh" "$LINK_TARGET"
        echo "kcadm.sh linked to $LINK_TARGET"
    else
        echo "kcadm.sh not found! Ensure Keycloak was installed correctly."
        exit 1
    fi
}

# Execute Steps
install_keycloak
setup_kcadm_link

echo "Installation complete! You can now use the kcadm command globally."

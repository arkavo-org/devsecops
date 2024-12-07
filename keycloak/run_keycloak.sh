# Check if the container is running
if docker ps --format '{{.Names}}' | grep -q "^keycloak\$"; then
  echo "Container 'keycloak' is already running. Exiting."
  exit 0
fi

# Check if the container exists but is not running
if docker ps -a --format '{{.Names}}' | grep -q "^keycloak\$"; then
  echo "Container 'keycloak' exists but is stopped. Starting it."
  docker start keycloak && echo "Container 'keycloak' started." || echo "Failed to start container 'keycloak'."
  exit 0
fi

# If the container does not exist, create and run it
echo "Container 'keycloak' does not exist. Creating and starting it."
docker network inspect $BRAND_NAME >/dev/null 2>&1 || docker network create $BRAND_NAME && \
docker run -d \
  --name keycloak \
  --network $BRAND_NAME \
  --restart always \
  -v $(pwd)/keys/localhost.crt:/etc/x509/tls/localhost.crt \
  -v $(pwd)/keys/localhost.key:/etc/x509/tls/localhost.key \
  -v $(pwd)/keys/ca.jks:/truststore/truststore.jks \
  -v $(pwd)/KCSetup:/opt/keycloak/KCSetup \
  -v $(pwd)/oauth_certs:/opt/keycloak/oauth_certs \
  -v $(pwd)/keycloak-startup.sh:/opt/keycloak/keycloak-startup.sh \
  -e KC_PROXY=edge \
  -e KC_HTTP_RELATIVE_PATH=/auth \
  -e KC_DB_VENDOR=postgres \
  -e KC_DB_URL_HOST=keycloakdb \
  -e KC_DB_URL_PORT=5432 \
  -e KC_DB_URL_DATABASE=keycloak \
  -e KC_DB_USERNAME=keycloak \
  -e KC_DB_PASSWORD=changeme \
  -e KC_HOSTNAME_STRICT=false \
  -e KC_HOSTNAME_STRICT_BACKCHANNEL=false \
  -e KC_HOSTNAME_STRICT_HTTPS=false \
  -e KC_HTTP_ENABLED=true \
  -e KC_HTTP_PORT=8888 \
  -e KC_HTTPS_PORT=8443 \
  -e KEYCLOAK_ADMIN=admin \
  -e KEYCLOAK_ADMIN_PASSWORD=changeme \
  -e KEYCLOAK_FRONTEND_URL=https://keycloak.juliancoy.us/auth \
  -e KC_HOSTNAME_URL=https://keycloak.juliancoy.us/auth \
  -e KC_FEATURES="preview,token-exchange" \
  -e KC_HEALTH_ENABLED=true \
  -e KC_HTTPS_KEY_STORE_PASSWORD=password \
  -e KC_HTTPS_KEY_STORE_FILE=/truststore/truststore.jks \
  -e KC_HTTPS_CERTIFICATE_FILE=/etc/x509/tls/localhost.crt \
  -e KC_HTTPS_CERTIFICATE_KEY_FILE=/etc/x509/tls/localhost.key \
  -e KC_HTTPS_CLIENT_AUTH=request \
  --entrypoint /bin/bash \
  -p 8888:8888 \
  -p 8443:8443 \
  cgr.dev/chainguard/keycloak@sha256:37895558d2e0e93ffff75da5900f9ae7e79ec6d1c390b18b2ecea6cee45ec26f /opt/keycloak/keycloak-startup.sh

if [ $? -eq 0 ]; then
  echo "Container 'keycloak' successfully created and started."
else
  echo "Failed to create and start container 'keycloak'."
  exit 1
fi

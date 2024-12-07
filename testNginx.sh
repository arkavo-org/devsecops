docker run --rm \
    -v ./nginx:/certs:ro \
    --network="codecollective" \
    curlimages/curl:latest \
    curl -v --cacert /certs/all-ca-certificates.crt https://nginx/keycloak/auth/realms/opentdf/.well-known/openid-configuration

docker run -it --rm \
    --entrypoint="bash" \
    -v "$(pwd)/admin_scripts:/app" \
    -e KEYCLOAK_ADMIN=admin \
    -e KEYCLOAK_ADMIN_PASSWORD=$KEYCLOAK_ADMIN_PASSWORD \
    --workdir="/app" \
    ubuntu:latest

#!/bin/bash

# Stop (down) OpenTDF service
docker compose -f docker-compose-opentdf.yaml down
echo "OpenTDF service has been stopped!"

# Stop (down) OpenTDF DB service
docker compose -f docker-compose-opentdfdb.yaml down
echo "OpenTDF DB service has been stopped!"

# Stop (down) Keycloak service
docker compose -f docker-compose-keycloak.yaml down
echo "Keycloak service has been stopped!"

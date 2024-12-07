#!/bin/bash

# Start Keycloak
./run_keycloak_db.sh

# make sure the postgres is up
docker run --rm --network $BRAND_NAME postgres:15-alpine sh -c '
  echo "Waiting for Keycloak DB to respond on keycloakdb:5432...";
  until pg_isready -h keycloakdbdb -p 5432 -U postgres >/dev/null 2>&1; do
    sleep 5;
    echo "Still waiting for Keycloak DB to accept connections on port 5432...";
  done;
  echo "Keycloak DB is accepting connections on keycloakdbdb:5432!";
'

./runKeycloak.sh

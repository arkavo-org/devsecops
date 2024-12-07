docker network inspect $BRAND_NAME >/dev/null 2>&1 || docker network create $BRAND_NAME && \
docker volume inspect keycloakdb_data >/dev/null 2>&1 || docker volume create keycloakdb_data && \

check_and_start_container "keycloakdb"

# Start the container if it doesn't exist
docker run -d \
--name keycloakdb \
--network $BRAND_NAME \
--restart always \
-e POSTGRES_PASSWORD=changeme \
-e POSTGRES_USER=postgres \
-e POSTGRES_DB=keycloak \
-v keycloakdb_data:/var/lib/postgresql/data \
--health-cmd "pg_isready" \
--health-interval 5s \
--health-timeout 5s \
--health-retries 10 \
postgres:15-alpine
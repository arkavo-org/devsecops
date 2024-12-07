# Ensure the network exists
docker network inspect $BRAND_NAME >/dev/null 2>&1 || docker network create $BRAND_NAME

# Run the container
docker run -d \
  --name opentdfdb \
  --network $BRAND_NAME \
  --restart always \
  --user postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=changeme \
  -e POSTGRES_DB=opentdf \
  -v opentdfdb_data:/var/lib/postgresql/data \
  --health-cmd="pg_isready" \
  --health-interval=5s \
  --health-timeout=5s \
  --health-retries=10 \
  postgres:15-alpine

# make sure the postgres is up
docker run --rm --network $BRAND_NAME postgres:15-alpine sh -c '
  echo "Waiting for OpenTDF DB to respond on opentdfdb:5432...";
  until pg_isready -h opentdfdb -p 5432 -U postgres >/dev/null 2>&1; do
    sleep 5;
    echo "Still waiting for OpenTDF DB to accept connections on port 5432...";
  done;
  echo "OpenTDF DB is accepting connections on opentdfdb:5432!";
'

# Run the container
docker run -d \
  --name opentdf \
  --network $BRAND_NAME \
  -p 8080:8080 \
  --restart always \
  -e KEYCLOAK_BASE_URL="${KEYCLOAK_SERVER_URL}" \
  -v "$(pwd)/opentdf-dev.yaml:/app/opentdf.yaml" \
  -v "$(pwd):/keys" \
  -v "$(pwd)/kas-cert.pem:/app/kas-cert.pem" \
  -v "$(pwd)/kas-ec-cert.pem:/app/kas-ec-cert.pem" \
  -v "$(pwd)/kas-private.pem:/app/kas-private.pem" \
  -v "$(pwd)/kas-ec-private.pem:/app/kas-ec-private.pem" \
  --health-cmd="curl -sf ${KEYCLOAK_SERVER_URL} || exit 1" \
  --health-interval=10s \
  --health-timeout=5s \
  --health-retries=5 \
  julianfl0w/opentdf:1.0 \
  start
  


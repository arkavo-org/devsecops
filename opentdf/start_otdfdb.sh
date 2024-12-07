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


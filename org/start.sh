# Ensure the network exists
docker network inspect $BRAND_NAME >/dev/null 2>&1 || docker network create $BRAND_NAME

# Run the container
docker run -d \
  --name org-backend \
  --network $BRAND_NAME \
  -p 8085:8085 \
  --restart always \
  -w /app \
  -v "$(pwd):/app" \
  -e KEYCLOAK_ADMIN="${KEYCLOAK_ADMIN}" \
  -e KEYCLOAK_ADMIN_PASSWORD="${KEYCLOAK_ADMIN_PASSWORD}" \
  -e PORT="${ORG_PORT:-8085}" \
  cgr.dev/chainguard/go:latest \
  run main.go


# Ensure the network exists
docker network inspect $BRAND_NAME >/dev/null 2>&1 || docker network create $BRAND_NAME

# Run the container
docker run -d \
  --name nginx-proxy \
  --network $BRAND_NAME \
  --rm \
  -v "$(pwd)/conf/nginx.conf:/etc/nginx/nginx.conf" \
  -v "$(pwd)/ssl:/etc/nginx/ssl" \
  -v "$(pwd)/html:/usr/share/nginx/html" \
  -v "/etc/letsencrypt/live/$BRAND_NAME$TLD/fullchain.pem:/keys/fullchain.pem" \
  -v "/etc/letsencrypt/live/$BRAND_NAME$TLD/privkey.pem:/keys/privkey.pem" \
  -p 80:80 \
  -p 443:443 \
  nginx:latest


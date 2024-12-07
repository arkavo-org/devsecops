certbot certonly --manual --preferred-challenges dns \
    --config-dir ./letsencrypt/config \
    --work-dir ./letsencrypt/work \
    --logs-dir ./letsencrypt/logs \
    --key-path ./privkey.pem \
    --fullchain-path ./fullchain.pem \
    --cert-path ./cert.pem


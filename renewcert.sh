docker stop nginx keycloak opentdf
certbot certonly --standalone --expand -d arkavo.org -d keycloak.arkavo.org -d opentdf.arkavo.org -d dev.arkavo.org -d matrix.arkavo.org
cp /etc/letsencrypt/live/arkavo.org/*.pem certs/
chmod 666 certs/*.pem
python run.py

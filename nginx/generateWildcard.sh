docker run --rm -it \
  -v /etc/letsencrypt:/etc/letsencrypt \
  certbot/certbot certonly \
  --manual --preferred-challenges dns \
  -d $BRAND_NAME.org -d *.$BRAND_NAME.org


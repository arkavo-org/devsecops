docker create --name opentdf_temp julianfl0w/opentdf:1.0
docker cp opentdf_temp:/etc/ssl/certs/ca-certificates.crt .
docker rm opentdf_temp

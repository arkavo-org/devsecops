#!/bin/bash

# Exit on any error
set -e

# Install OpenSSL (for Alpine-based systems)
apk add --no-cache openssl

# Create certs directory
mkdir -p /certs

# Write OpenSSL config to a temporary file
cat <<EOF > /tmp/openssl.cnf
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = req_ext

[dn]
CN = nginx

[req_ext]
subjectAltName = @alt_names

[alt_names]
DNS.1 = nginx
DNS.2 = localhost
IP.1 = 127.0.0.1
EOF

# Create CA key
openssl genrsa -out /certs/ca.key 2048

# Create CA certificate (self-signed)
openssl req -x509 -new -nodes -key /certs/ca.key -sha256 -days 3650 \
    -subj "/C=US/ST=CA/L=Local/O=MyOrg/CN=MyCA" -out /certs/ca.crt

# Create server key
openssl genrsa -out /certs/privkey.pem 2048

# Create CSR for server certificate using the SAN config
openssl req -new -key /certs/privkey.pem -config /tmp/openssl.cnf -out /tmp/server.csr

# Sign the server CSR with the CA key and certificate
openssl x509 -req -in /tmp/server.csr -CA /certs/ca.crt -CAkey /certs/ca.key -CAcreateserial \
    -days 365 -sha256 -extfile /tmp/openssl.cnf -extensions req_ext -out /certs/server.crt

# Combine server cert and CA cert into a full chain
cat /certs/server.crt /certs/ca.crt > /certs/fullchain.pem

# Set permissions
chmod 644 /certs/privkey.pem /certs/server.crt /certs/fullchain.pem /certs/ca.crt

# Combine the certificates with the standard trust store
cat /etc/ssl/certs/ca-certificates.crt /certs/ca.crt /certs/server.crt > /certs/all-ca-certificates.crt

echo "Certificates generated and stored in /certs"

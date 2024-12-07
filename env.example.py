import os
import requests
import json 

current_dir = os.path.abspath(os.path.dirname(__file__))
keycloak_dir = os.path.join(current_dir, "keycloak")
opentdf_dir = os.path.join(current_dir, "opentdf")
nginx_dir = os.path.join(current_dir, "nginx")
webapp_dir = os.path.join(current_dir, "webapp")

# Most common options to change
BRAND_NAME = "yourbrand"
USER_WEBSITE = "localhost"
PROTOCOL_USER_WEBSITE = "https://localhost"
TLD = ".us"
LOCAL_SERVER_mDNS = "localhost"
SERVICES_TO_RUN = ["keycloak", "org", "opentdf", "AICouncil", "nginx"]


# Check to see if we're in an EC2 instance
ec2_metadata_base_url = "http://169.254.169.254/latest/meta-data/"
try:
    response = requests.get(ec2_metadata_base_url, timeout=1)
    response.raise_for_status()
    metadata_keys = response.text.splitlines()

    metadata = {}
    for key in metadata_keys:
        key_url = ec2_metadata_base_url + key
        key_response = requests.get(key_url, timeout=1)
        key_response.raise_for_status()
        metadata[key] = key_response.text
    
    print(json.dumps(metadata))
    IS_EC2 = True

except requests.RequestException as e:
    print("No EC2 Metadata. Assuming local deployment")
    IS_EC2 = False

# -- Locations of Services -- 
if IS_EC2:
    pass
else:
    KEYCLOAK_BASE_URL = "localhost/keycloak"
    VITE_PUBLIC_URL = "localhost"

# Git Branch Port Config
# BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Keycloak Config
KEYCLOAK_REALM = "opentdf"
KEYCLOAK_PROTOCOL = "https"
KEYCLOAK_PORT = "8888" # if applicable
KEYCLOAK_INTERNAL_CHECK_ADDR = "http://keycloak:8888"
KEYCLOAK_INTERNAL_AUTH_URL = "https://keycloak:8443/keycloak/auth"
KEYCLOAK_HOST = KEYCLOAK_PROTOCOL + "://" + KEYCLOAK_BASE_URL
KEYCLOAK_AUTH_URL = KEYCLOAK_HOST + "/auth"
VITE_KEYCLOAK_AUTH_ENDPOINT = (
    f"{KEYCLOAK_HOST}/auth/realms/{KEYCLOAK_REALM}/protocol/openid-connect/auth"
)
VITE_KEYCLOAK_TOKEN_ENDPOINT = (
    f"{KEYCLOAK_HOST}/auth/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"
)
VITE_KEYCLOAK_USERINFO_ENDPOINT = (
    f"{KEYCLOAK_HOST}/auth/realms/{KEYCLOAK_REALM}/protocol/openid-connect/userinfo"
)
VITE_KEYCLOAK_SERVER_URL = KEYCLOAK_HOST + "/auth"
VITE_KEYCLOAK_CLIENT_ID = BRAND_NAME
VITE_KEYCLOAK_REALM = KEYCLOAK_REALM
VITE_KAS_ENDPOINT = "https://localhost/opentdf/kas"
KEYCLOAK_ADMIN_PASSWORD = "changeme" # Secrets

# Other ish
# Google OAuth Config
GOOGLE_CLIENT_SECRET = "<YOUR SECRET HERE>"
VITE_GOOGLE_SCOPES = "openid profile email"
VITE_GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
VITE_GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
VITE_GOOGLE_CLIENT_ID = "<YOUR GOOGLE OAUTH CLIENT ID>"

# GitHub OAuth Config
GITHUB_CLIENT_SECRET = "<YOUR SECRET HERE>"
VITE_GITHUB_CLIENT_ID = "<YOUR GITHUB OAUTH CLIENT ID>"
VITE_GITHUB_SCOPES = "openid profile email"
VITE_GITHUB_AUTH_ENDPOINT = "https://github.com/login/oauth/authorize"
VITE_GITHUB_TOKEN_ENDPOINT = "https://github.com/login/oauth/access_token"

PEM_FILE = "<YOUR SECRET HERE>"
SERVER_USER = "<YOUR SECRET HERE>"
SERVER_HOST = "<YOUR SECRET HERE>"
REMOTE_FOLDER = "<YOUR SECRET HERE>"
ZIP_FILE = "<YOUR SECRET HERE>"
LOCAL_DESTINATION = "<YOUR SECRET HERE>"
EXTRACT_FOLDER = "<YOUR SECRET HERE>"  # Name of the folder after extraction

# More public options
COMPOSE_PROJECT_NAME = BRAND_NAME

# Admin Config
ADMIN_CLIENT = "admin-cli"
KEYCLOAK_ADMIN = "admin"

# EC2 Metadata Check
VITE_PUBLIC_URL_ON_EC2 = "localhost"

# Redirects
VITE_GOOGLE_REDIRECT_URI = f"https://{VITE_PUBLIC_URL}"
VITE_GITHUB_REDIRECT_URI = f"{VITE_PUBLIC_URL}"
VITE_KEYCLOAK_REDIRECT_URI = f"{VITE_PUBLIC_URL}"
KEYCLOAK_REDIRECT_URI = f"{VITE_KEYCLOAK_REDIRECT_URI}"
VITE_ORG_BACKEND_URL = "https://localhost/org"
VITE_FRONTEND_SERVER_URL = f"https://{VITE_PUBLIC_URL}"

# Opentdf config
opentdfdb = dict(
    image="postgres:15-alpine",
    detach=True,
    name="opentdfdb",
    network=BRAND_NAME,
    restart_policy={"Name": "always"},
    user="postgres",
    environment={
        "POSTGRES_PASSWORD": "changeme",
        "POSTGRES_USER": "postgres",
        "POSTGRES_DB": "opentdf",
    },
    volumes={
        "POSTGRES_DATA_VOLUME": {"bind": "/var/lib/postgresql/data", "mode": "rw"}
    },
    healthcheck={
        "test": ["CMD-SHELL", "pg_isready"],
        "interval": 5000000000,  # 5s in nanoseconds
        "timeout": 5000000000,  # 5s in nanoseconds
        "retries": 10,
    },
)

opentdf = dict(
    image="julianfl0w/opentdf:1.0",
    detach=True,
    command="start",
    name="opentdf",
    network=BRAND_NAME,
    restart_policy={"Name": "always"},
    ports={"8080/tcp": 8080},
    environment={
        #"KEYCLOAK_BASE_URL": KEYCLOAK_AUTH_URL,
        "KEYCLOAK_BASE_URL": KEYCLOAK_INTERNAL_AUTH_URL,
    },
    volumes={
        f"{nginx_dir}/all-ca-certificates.crt": {"bind": "/etc/ssl/certs/ca-certificates.crt", "mode": "ro"},
        f"{opentdf_dir}/opentdf.yaml": {"bind": "/app/opentdf.yaml", "mode": "ro"},
        f"{keycloak_dir}/kas-cert.pem": {"bind": "/keys/kas-cert.pem", "mode": "ro"},
        f"{keycloak_dir}/kas-ec-cert.pem": {"bind": "/keys/kas-ec-cert.pem", "mode": "ro"},
        f"{keycloak_dir}/kas-private.pem": {"bind": "/keys/kas-private.pem", "mode": "ro"},
        f"{keycloak_dir}/kas-ec-private.pem": {"bind": "/keys/kas-ec-private.pem", "mode": "ro"},

        # Mount the CA key from nginx directory
        f"{nginx_dir}/ca.key": {"bind": "/app/nginx/ca.key", "mode": "ro"}
    },
    healthcheck={
        "test": ["CMD-SHELL", f"curl -sf {KEYCLOAK_AUTH_URL} || exit 1"],
        "interval": 10000000000,  # 10s in nanoseconds
        "timeout": 5000000000,    # 5s in nanoseconds
        "retries": 5,
    },
)


# Keycloak config
keycloakdb = opentdfdb.copy()
keycloakdb["name"] = "keycloakdb"

keycloak = {
    "name": "keycloak",
    "network": BRAND_NAME,
    "image": "cgr.dev/chainguard/keycloak@sha256:37895558d2e0e93ffff75da5900f9ae7e79ec6d1c390b18b2ecea6cee45ec26f",
    "entrypoint": "/opt/keycloak/keycloak-startup.sh",
    "detach": True,
    "volumes": {
        os.path.join(keycloak_dir, "keys/localhost.crt"): {
            "bind": "/etc/x509/tls/localhost.crt",
            "mode": "ro",  # Read-only
        },
        os.path.join(keycloak_dir, "keys/localhost.key"): {
            "bind": "/etc/x509/tls/localhost.key",
            "mode": "ro",  # Read-only
        },
        os.path.join(keycloak_dir, "keys/ca.jks"): {
            "bind": "/truststore/truststore.jks",
            "mode": "ro",  # Read-only
        },
        os.path.join(keycloak_dir, "realms"): {
            "bind": "/opt/keycloak/realms",
            "mode": "ro",  # Read-only
        },
        os.path.join(keycloak_dir, "oauth_certs"): {
            "bind": "/opt/keycloak/oauth_certs",
            "mode": "ro",  # Read-only
        },
        os.path.join(keycloak_dir, "keycloak-startup.sh"): {
            "bind": "/opt/keycloak/keycloak-startup.sh",
            "mode": "ro",  # Read-only
        },
    },
    "environment": {
        "KC_PROXY": "edge",
        "PROXY_ADDRESS_FORWARDING": "true",
        "KC_HTTP_RELATIVE_PATH": "/keycloak/auth",
        "KC_DB_VENDOR": "postgres",
        "KC_DB_URL_HOST": "keycloakdb",
        "KC_DB_URL_PORT": "5432",
        "KC_DB_URL_DATABASE": "keycloak",
        "KC_DB_USERNAME": "keycloak",
        "KC_DB_PASSWORD": "changeme",
        "KC_HOSTNAME_STRICT": "false",
        "KC_HOSTNAME_STRICT_BACKCHANNEL": "false",
        "KC_HOSTNAME_STRICT_HTTPS": "false",
        "KC_HTTP_ENABLED": "true",
        "KC_HTTP_PORT": "8888",
        "KC_HTTPS_PORT": "8443",
        "KEYCLOAK_ADMIN": "admin",
        "KEYCLOAK_ADMIN_PASSWORD": "changeme",
        "KEYCLOAK_FRONTEND_URL": KEYCLOAK_AUTH_URL,
        "KC_HOSTNAME_URL": KEYCLOAK_AUTH_URL,
        "KC_FEATURES": "preview,token-exchange",
        "KC_HEALTH_ENABLED": "true",
        "KC_HTTPS_KEY_STORE_PASSWORD": "password",
        "KC_HTTPS_KEY_STORE_FILE": "/truststore/truststore.jks",
        "KC_HTTPS_CERTIFICATE_FILE": "/etc/x509/tls/localhost.crt",
        "KC_HTTPS_CERTIFICATE_KEY_FILE": "/etc/x509/tls/localhost.key",
        "KC_HTTPS_CLIENT_AUTH": "request",
    },
}

nginx = dict(
    image='nginx:latest',
    name='nginx',
    detach=True,  # equivalent to -d
    network=BRAND_NAME,  # equivalent to --network $BRAND_NAME
    remove=True,   # equivalent to --rm
    volumes={
        os.path.join(nginx_dir, 'conf', 'nginx.conf'): {
            'bind': '/etc/nginx/nginx.conf',
            'mode': 'rw'
        },
        os.path.join(nginx_dir, 'ssl'): {
            'bind': '/etc/nginx/ssl',
            'mode': 'rw'
        },
        os.path.join(nginx_dir, 'html'): {
            'bind': '/usr/share/nginx/html',
            'mode': 'rw'
        },
        f"{nginx_dir}/fullchain.pem": {
            'bind': '/keys/fullchain.pem',
            'mode': 'rw'
        },
        f"{nginx_dir}/privkey.pem": {
            'bind': '/keys/privkey.pem',
            'mode': 'rw'
        }
    },
    ports={
        '80/tcp': 80,   # equivalent to -p 80:80
        '443/tcp': 443  # equivalent to -p 443:443
    }
)

webapp = dict(
    image="node:22",
    detach=True,  # Runs the container in detached mode
    name=f"webapp",
    network=BRAND_NAME,
    remove=True,  # Automatically removes the container when stopped
    volumes={
        webapp_dir: {"bind": "/usr/src/app", "mode": "rw"}
    },
    working_dir="/usr/src/app",
    ports={
        "3001": "5173"
    },
    environment={
        "NODE_ENV": "development",
        "VITE_KEYCLOAK_SERVER_URL": VITE_KEYCLOAK_SERVER_URL,
        "VITE_KEYCLOAK_REALM": VITE_KEYCLOAK_REALM,
        "VITE_KEYCLOAK_CLIENT_ID": VITE_KEYCLOAK_CLIENT_ID,
        "VITE_ORG_BACKEND_URL": VITE_ORG_BACKEND_URL,
        "VITE_KAS_ENDPOINT": VITE_KAS_ENDPOINT,
    },
    command="sh -c 'npm install && npm run dev'",
)

from editme import *

# The remainder of the environment can be generated
import os
import requests
import json
import copy
import util

# Get the current user's UID and GID
uid = os.getuid()
gid = os.getgid()

# Determine the absolute paths of salient directories
current_dir = os.getenv("CURRENT_DIR", os.path.abspath(os.path.dirname(__file__)))
keycloak_dir = os.path.join(current_dir, "keycloak")
opentdf_dir = os.path.join(current_dir, "opentdf")
nginx_dir = os.path.join(current_dir, "nginx")
webapp_dir = os.path.join(current_dir, "webapp")
org_dir = os.path.join(current_dir, "org")
certs_dir = os.path.join(current_dir, "certs")
keys_dir = os.path.join(certs_dir, "keys")
synapse_dir = os.path.join(current_dir, "synapse")
bsky_bridge_dir = os.path.join(current_dir, "bsky_bridge")
gitea_dir = os.path.join(current_dir, "gitea")
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
    print("Detected EC2 Runtime")

except requests.RequestException as e:
    print("No EC2 Metadata. Assuming local deployment")
    IS_EC2 = False


MODELS_TO_PULL = [
    "llama3.2",
    "ALIENTELLIGENCE/sigmundfreud",
    "phi3",
    "deepseek-coder:6.7b",  # "omercelik/mistral-small-coder
]

# Keycloak Addresses
KEYCLOAK_BASE_URL = "keycloak." + BACKEND_LOCATION
OPENTDF_BASE_URL = "opentdf." + BACKEND_LOCATION
ORG_BASE_URL = "org." + BACKEND_LOCATION
SYNAPSE_BASE_URL = "matrix." + BACKEND_LOCATION
BLUESKY_BASE_URL = "bluesky." + BACKEND_LOCATION
BSKY_FYP_BASE_URL = "bsky_fyp." + BACKEND_LOCATION
ELEMENT_BASE_URL = "element." + BACKEND_LOCATION
BSKY_BRIDGE_BASE_URL = "bsky_bridge." + BACKEND_LOCATION
WEBAPP_DEV_BASE_URL = "dev." + USER_WEBSITE
INITIATIVE_BASE_URL = "initiative." + BACKEND_LOCATION

KEYCLOAK_HOST = "https://" + KEYCLOAK_BASE_URL
PROTOCOL_OPENTDF_BASE_URL = "https://" + OPENTDF_BASE_URL
PROTOCOL_ORG_BASE_URL = "https://" + ORG_BASE_URL
PROTOCOL_SYNAPSE_BASE_URL = "https://" + SYNAPSE_BASE_URL
VITE_SYNAPSE_BASE_URL = SYNAPSE_BASE_URL
VITE_PROTOCOL_SYNAPSE_BASE_URL = PROTOCOL_SYNAPSE_BASE_URL

VITE_BLUESKY_HOST = "https://" + BLUESKY_BASE_URL
VITE_PUBLIC_URL = USER_WEBSITE

# GPU supported services
DEEPSEEK_JANUS_BASE_URL = "deepseek_janus." + GPU_BACKEND_LOCATION
OLLAMA_BASE_URL = "ollama." + GPU_BACKEND_LOCATION

# Git Branch Port Config
# BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Keycloak Config
KEYCLOAK_REALM = "opentdf"
KEYCLOAK_INTERNAL_CHECK_ADDR = f"https://{KEYCLOAK_INTERNAL_URL}"
KEYCLOAK_INTERNAL_AUTH_URL = f"{KEYCLOAK_HOST}/auth"
VITE_KEYCLOAK_SERVER_URL = KEYCLOAK_HOST + "/auth"

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
KEYCLOAK_SERVER_URL_INTERNAL = "https://keycloak:8443/auth"
VITE_KEYCLOAK_CLIENT_ID = "web-client"
VITE_KEYCLOAK_REALM = KEYCLOAK_REALM
VITE_KAS_ENDPOINT = f"https://{OPENTDF_BASE_URL}/kas"

# More public options
NETWORK_NAME = BRAND_NAME + distinguisher

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
    name="opentdfdb" + distinguisher,
    network=NETWORK_NAME,
    restart_policy={"Name": "always"},
    user="postgres",
    environment={
        "POSTGRES_PASSWORD": "changeme",
        "POSTGRES_USER": "postgres",
        "POSTGRES_DB": "opentdf",
    },
    volumes={
        "OPENTDF_POSTGRES"
        + distinguisher: {
            "bind": "/var/lib/postgresql/data",
            "mode": "rw",
        }
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
    network=NETWORK_NAME,
    restart_policy={"Name": "always"},
    environment={
        "KEYCLOAK_BASE_URL": KEYCLOAK_INTERNAL_AUTH_URL,
    },
    volumes={
        # f"{certs_dir}/all-ca-certificates.crt": {"bind": "/etc/ssl/certs/ca-certificates.crt", "mode": "ro"},
        f"{opentdf_dir}/opentdf.yaml": {"bind": "/app/opentdf.yaml", "mode": "ro"},
        f"{keys_dir}/kas-cert.pem": {"bind": "/keys/kas-cert.pem", "mode": "ro"},
        f"{keys_dir}/kas-ec-cert.pem": {"bind": "/keys/kas-ec-cert.pem", "mode": "ro"},
        f"{keys_dir}/kas-private.pem": {"bind": "/keys/kas-private.pem", "mode": "ro"},
        f"{keys_dir}/kas-ec-private.pem": {
            "bind": "/keys/kas-ec-private.pem",
            "mode": "ro",
        },
        # f"{keys_dir}/keycloak-ca.pem": {"bind": "/etc/ssl/certs/ca-certificates.crt", "mode": "ro"},
        f"{keys_dir}/keycloak-ca.pem": {
            "bind": "/usr/local/share/ca-certificates/ca-certificates.crt",
            "mode": "ro",
        },
        # Mount the CA key from nginx directory
        # f"{certs_dir}/ca.key": {"bind": "/app/nginx/ca.key", "mode": "ro"}
    },
    healthcheck={
        "test": ["CMD-SHELL", f"curl -sf {KEYCLOAK_AUTH_URL} || exit 1"],
        "interval": 10000000000,  # 10s in nanoseconds
        "timeout": 5000000000,  # 5s in nanoseconds
        "retries": 5,
    },
)


# Keycloak config
keycloakdb = copy.deepcopy(opentdfdb)
keycloakdb["name"] = "keycloakdb"
keycloakdb["environment"]["POSTGRES_DB"] = "keycloak"
keycloakdb["volumes"] = {
    "KEYCLOAK_POSTGRES"
    + distinguisher: {"bind": "/var/lib/postgresql/data", "mode": "rw"}
}

keycloak = {
    "name": "keycloak",
    "network": NETWORK_NAME,
    #"image": "cgr.dev/chainguard/keycloak@sha256:37895558d2e0e93ffff75da5900f9ae7e79ec6d1c390b18b2ecea6cee45ec26f",
    "image": "quay.io/keycloak/keycloak",
    "entrypoint": "/opt/keycloak/keycloak-startup.sh",
    "detach": True,
    "restart_policy": {"Name": "always"},
    "volumes": {
        os.path.join(keys_dir, "localhost.crt"): {
            "bind": "/etc/x509/tls/localhost.crt",
            "mode": "ro",  # Read-only
        },
        os.path.join(keys_dir, "localhost.key"): {
            "bind": "/etc/x509/tls/localhost.key",
            "mode": "ro",  # Read-only
        },
        os.path.join(keys_dir, "ca.jks"): {
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
        os.path.join(keycloak_dir, "theme"): {
            "bind": "/opt/keycloak/themes/arkavo",
            "mode": "rw",  # Read-only
        },
    },
    "environment": {
        "KC_PROXY": "edge",
        "PROXY_ADDRESS_FORWARDING": "true",
        "KC_HTTP_RELATIVE_PATH": "/auth",
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
        "KEYCLOAK_ADMIN": KEYCLOAK_ADMIN,
        "KEYCLOAK_ADMIN_PASSWORD": KEYCLOAK_ADMIN_PASSWORD,
        "KEYCLOAK_FRONTEND_URL": KEYCLOAK_AUTH_URL,
        "KC_HOSTNAME_URL": KEYCLOAK_AUTH_URL,
        "KC_FEATURES": "preview,token-exchange",
        "KC_LOG_LEVEL": "INFO",
        "KC_HEALTH_ENABLED": "true",
        "KC_HTTPS_KEY_STORE_PASSWORD": "password",
        "KC_HTTPS_KEY_STORE_FILE": "/truststore/truststore.jks",
        "KC_HTTPS_CERTIFICATE_FILE": "/etc/x509/tls/localhost.crt",
        "KC_HTTPS_CERTIFICATE_KEY_FILE": "/etc/x509/tls/localhost.key",
        "KC_HTTPS_CLIENT_AUTH": "request",
        "KC_SPI_THEME_STATIC_MAX_AGE": -1,
        "KC_SPI_THEME_CACHE_THEMES": False,
        "KC_SPI_THEME_CACHE_TEMPLATES": False,
    },
}

nginx = dict(
    image="nginx:latest",
    name="nginx",
    detach=True,  # equivalent to -d
    network=NETWORK_NAME,
    restart_policy={"Name": "always"},
    volumes={
        os.path.join(nginx_dir, "nginx.conf"): {
            "bind": "/etc/nginx/nginx.conf",
            "mode": "rw",
        },
        os.path.join(webapp_dir, "dist"): {
            "bind": "/app",
            "mode": "rw",
        },
        os.path.join(certs_dir, "ssl"): {"bind": "/etc/nginx/ssl", "mode": "rw"},
        os.path.join(certs_dir, "html"): {
            "bind": "/usr/share/nginx/html",
            "mode": "rw",
        },
        f"{certs_dir}/fullchain.pem": {"bind": "/keys/fullchain.pem", "mode": "rw"},
        f"{certs_dir}/privkey.pem": {"bind": "/keys/privkey.pem", "mode": "rw"},
    },
    ports={
        "80/tcp": 80,  # equivalent to -p 80:80
        "443/tcp": 443,  # equivalent to -p 443:443
        "6667/tcp": 6667,
        "8443/tcp": 8443,
        "8448/tcp": 8448,
    },
)

webapp_build = dict(
    image="node:23",
    detach=True,  # Runs the container in detached mode
    name=f"webapp_build",
    network=NETWORK_NAME,
    restart_policy={"Name": "no"},
    volumes={
        webapp_dir: {"bind": "/usr/src/app", "mode": "rw"},
        # "dist_volume": {"bind": "/usr/src/app/dist", "mode": "rw"},
    },
    working_dir="/usr/src/app",
    environment={
        "NODE_ENV": "development",
        "VITE_KEYCLOAK_SERVER_URL": VITE_KEYCLOAK_SERVER_URL,
        "VITE_KEYCLOAK_REALM": VITE_KEYCLOAK_REALM,
        "VITE_KEYCLOAK_CLIENT_ID": VITE_KEYCLOAK_CLIENT_ID,
        "VITE_ORG_BACKEND_URL": VITE_ORG_BACKEND_URL,
        "VITE_KAS_ENDPOINT": VITE_KAS_ENDPOINT,
    },
    command="sh -c 'npm install && npm run build  --verbose'",
    user=f"{uid}:{gid}",
)

webapp = dict(
    image="node:23",
    detach=True,  # Runs the container in detached mode
    name=f"webapp",
    network=NETWORK_NAME,
    restart_policy={"Name": "always"},
    volumes={
        webapp_dir: {"bind": "/usr/src/app", "mode": "rw"},
        # "dist_volume": {"bind": "/usr/src/app/dist", "mode": "rw"},
    },
    working_dir="/usr/src/app",
    environment={
        "NODE_ENV": "development",
        "VITE_KEYCLOAK_SERVER_URL": VITE_KEYCLOAK_SERVER_URL,
        "VITE_KEYCLOAK_REALM": VITE_KEYCLOAK_REALM,
        "VITE_KEYCLOAK_CLIENT_ID": VITE_KEYCLOAK_CLIENT_ID,
        "VITE_ORG_BACKEND_URL": VITE_ORG_BACKEND_URL,
        "VITE_KAS_ENDPOINT": VITE_KAS_ENDPOINT,
    },
    command=(
        'sh -c "'
        "npm install && "
        "npm install -g nodemon && "
        "nodemon --watch . --exec 'npm run dev'\""
    ),
    # user=uid,
    # group_add=[gid],
)

go_installs_dir = os.path.join(
    org_dir, "installs"
)  # Directory to hold Go installs on the host

org = dict(
    image="cosmtrek/air:latest",
    detach=True,  # Runs the container in detached mode
    name=f"org",
    network=NETWORK_NAME,
    restart_policy={"Name": "always"},
    volumes={
        org_dir: {"bind": "/usr/src/app", "mode": "rw"},
        go_installs_dir: {
            "bind": "/go/pkg/mod",
            "mode": "rw",
        },  # Volume for Go installs
        certs_dir: {"bind": "/certs", "mode": "rw"},
    },
    working_dir="/usr/src/app",
    environment={
        "ORG_BACKEND_URL": VITE_ORG_BACKEND_URL,
        "FRONTEND_URL": USER_WEBSITE,
        "ALLOWED_ORIGINS": f"{USER_WEBSITE},https://arkavo.ai",
        "KEYCLOAK_ADMIN": KEYCLOAK_ADMIN,
        "KEYCLOAK_ADMIN_PASSWORD": KEYCLOAK_ADMIN_PASSWORD,
        "KEYCLOAK_SERVER_URL": KEYCLOAK_INTERNAL_AUTH_URL,
        "BLUESKY_PDS_URL": VITE_BLUESKY_HOST,
        "ENCRYPTION_KEY": "temporary-key-please-change",
    },
    command=["sh", "-c", "go build && ./main"],
)

synapse = dict(
    image="matrixdotorg/synapse:latest",
    detach=True,
    name="synapse",
    network=NETWORK_NAME,
    restart_policy={"Name": "always"},
    volumes={
        os.path.join(current_dir, "synapse"): {"bind": "/data", "mode": "rw"},
        os.path.join(current_dir, "synapse", "templates"): {
            "bind": "/usr/local/lib/python3.12/site-packages/synapse/res/templates",
            "mode": "rw",
        },
    },
    healthcheck={
        "test": ["CMD-SHELL", "curl -f http://localhost:8008/health || exit 1"],
        "interval": 5000000000,  # 5s
        "timeout": 5000000000,  # 5s
        "retries": 5,
    },
)

synapsedb = copy.deepcopy(opentdfdb)
synapsedb["name"] = "synapsedb"
synapsedb["environment"]["POSTGRES_DB"] = "synapse"
synapsedb["environment"][
    "POSTGRES_INITDB_ARGS"
] = "--encoding=UTF8 --lc-collate=C --lc-ctype=C"
synapsedb["volumes"] = {
    "SYNAPSE_POSTGRES"
    + distinguisher: {"bind": "/var/lib/postgresql/data", "mode": "rw"}
}


# Base configuration
ollama = {
    "name": "ollama",
    "ports": {"11434/tcp": 11434},
    "network": NETWORK_NAME,
    "detach": True,  # Runs the container in detached mode
    "volumes": {
        os.path.join(current_dir, "ollama", "ollama_models"): {
            "bind": "/root/.ollama/models",
            "mode": "rw",
        },
        os.path.join(current_dir, "ollama", "whisper_models"): {
            "bind": "/data/cache/whisper/models",
            "mode": "rw",
        },
    },
    "environment": {
        "OLLAMA_ORIGINS": "*",
        "ENABLE_OLLAMA_API": "True",
        "DATA_DIR": "/data",
    },
    "image": "ollama/ollama",
}

# Base configuration
deepseek_janus = {
    "name": "deepseek_janus",
    "ports": {"8000/tcp": 8000},
    "network": NETWORK_NAME,
    "detach": True,  # Runs the container in detached mode
    "volumes": {
        os.path.join(current_dir, "huggingface"): {
            "bind": "/root/.cache/huggingface",
            "mode": "rw",
        },
    },
    "environment": {
        "MODEL_NAME": "deepseek-ai/Janus-Pro-1B",
    },
    "image": "julianfl0w/janus:latest",
}

sglang = dict(
    image="lmsysorg/sglang:latest",
    name="sglang",
    volumes={
        os.path.join(current_dir, "huggingface"): {
            "bind": "/root/.cache/huggingface",
            "mode": "rw",
        },
    },
    restart_policy={"Name": "always"},
    detach=True,
    network=NETWORK_NAME,
    # Uncomment if using port mapping instead of host mode
    # "ports": {"30000/tcp": 30000},
    environment={
        "HF_TOKEN": "<secret>",
        # Uncomment if using modelscope
        # "SGLANG_USE_MODELSCOPE": "true"
    },
    entrypoint="python3 -m sglang.launch_server",
    command=[
        "--model-path",
        "meta-llama/Llama-3.1-8B-Instruct",
        "--host",
        "0.0.0.0",
        "--port",
        "30000",
    ],
    ulimits=[
        {"Name": "memlock", "Soft": -1, "Hard": -1},
        {"Name": "stack", "Soft": 67108864, "Hard": 67108864},
    ],
    ipc_mode="host",
    healthcheck={
        "test": ["CMD-SHELL", "curl -f http://localhost:30000/health || exit 1"]
    },
)

from docker.types import DeviceRequest

# Check for NVIDIA GPU
if util.check_nvidia_gpu():
    drequests = DeviceRequest(count=1, capabilities=[["gpu"]], driver="nvidia")
    ollama["device_requests"] = [drequests]
    sglang["device_requests"] = [drequests]
    deepseek_janus["device_requests"] = [drequests]

# Check for AMD GPU
elif util.check_amd_gpu():
    drequests = DeviceRequest(count=1, capabilities=[["gpu"]], driver="amd")
    ollama["device_requests"] = [drequests]
    sglang["device_requests"] = [drequests]
    deepseek_janus["device_requests"] = [drequests]

# BLUESKY CRYPTO SETUP
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
import secrets

# Generate a secp256k1 private key
private_key = ec.generate_private_key(ec.SECP256K1())
private_key_bytes = private_key.private_numbers().private_value.to_bytes(
    32, byteorder="big"
)

if os.path.exists("jwt_secret.txt"):
    with open("jwt_secret.txt", "r") as file:
        JWT_SECRET = file.read()
else:
    JWT_SECRET = secrets.token_hex(16)
    with open("jwt_secret.txt", "w+") as file:
        file.write(JWT_SECRET)

element = dict(
    image="vectorim/element-web:latest",
    name="element",
    detach=True,  # Runs the container in detached mode
    restart_policy={"Name": "unless-stopped"},
    volumes={
        f"{synapse_dir}/element-config.json": {
            "bind": "/app/config.json",
            "mode": "rw",
        },
    },
    network=NETWORK_NAME,
)

bluesky_bridge = dict(
    image="python:3.11-slim",
    detach=True,
    name="bsky_bridge",
    working_dir="/app",
    network=NETWORK_NAME,
    volumes={
        bsky_bridge_dir: {"bind": "/app", "mode": "rw"},
    },
    environment=dict(
        BLUESKY_HANDLE=BLUESKY_HANDLE,
        BLUESKY_PASSWORD=BLUESKY_PASSWORD,
    ),
    command=["sh", "-c", "pip install atproto flask && python serve_feed.py"],
)

bsky_fyp = copy.copy(bluesky_bridge)
bsky_fyp["name"] = "bsky_fyp"
bsky_fyp["command"] = [
    "sh",
    "-c",
    "pip install atproto flask && python serve_vertical_fyp.py",
]

bluesky = dict(
    image="ghcr.io/bluesky-social/pds:latest",
    detach=True,
    name="pds",
    network=NETWORK_NAME,  # Make sure it's on the same network as nginx
    volumes={
        "bluesky_pds": {"bind": "/pds", "mode": "rw"},
    },
    environment=dict(
        DEBUG=1,
        PDS_HOSTNAME=USER_WEBSITE,
        PDS_JWT_SECRET=JWT_SECRET,
        ADMIN_HANDLE="admin",
        ADMIN_USERNAME="admin",
        PDS_ADMIN_PASSWORD="changeme",
        PDS_PLC_ROTATION_KEY_K256_PRIVATE_KEY_HEX=private_key_bytes.hex(),
        PDS_DATA_DIRECTORY="/pds",
        PDS_BLOBSTORE_DISK_LOCATION="/pds/blocks",
        PDS_BLOB_UPLOAD_LIMIT=52428800,
        PDS_DID_PLC_URL="https://plc.directory",
        PDS_BSKY_APP_VIEW_URL="https://api.bsky.app",
        PDS_BSKY_APP_VIEW_DID="did:web:api.bsky.app",
        PDS_REPORT_SERVICE_URL="https://mod.bsky.app",
        PDS_REPORT_SERVICE_DID="did:plc:ar7c4by46qjdydhdevvrndac",
        PDS_CRAWLERS="https://bsky.network",
        LOG_ENABLED="true",
    ),
)


gitea = dict(
    image="gitea/gitea:latest",
    detach=True,
    network=NETWORK_NAME,
    environment={"USER_UID": "1000", "USER_GID": "1000", "ACTIONS_ENABLED": "true"},
    volumes={
        gitea_dir: {"bind": "/data", "mode": "rw"},
    },
    restart_policy={"Name": "unless-stopped"},
)


act_runner = dict(
    image="gitea/act_runner:latest",
    detach=True,
    network=NETWORK_NAME,
    restart="always",
    depends_on=["gitea"],
    volumes=["/var/run/docker.sock:/var/run/docker.sock", "./act_runner:/data"],
    environment={
        "GITEA_INSTANCE_URL": "http://gitea:3000",
        "RUNNER_LABELS": "ubuntu-latest:docker://node:16-bullseye",
    },
)

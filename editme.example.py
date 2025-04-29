# Most common options to change
BACKEND_LOCATION = "arkavo.org"
GPU_BACKEND_LOCATION = "app.codecollective.us"
BRAND_NAME = "arkavo"
VITE_BRAND_NAME = "arkavo"
BRAND_COLOR_DARK = "#f99742"
BRAND_COLOR_LIGHT = "#F4A460"
USER_WEBSITE = "localhost"
PROTOCOL_USER_WEBSITE = "https://" + USER_WEBSITE
USER_EMAIL = "youremail@example.com"
VITE_LOGO_URL="/favicon.png"
KEYCLOAK_ADMIN_PASSWORD = "changeme"
SERVICES_TO_RUN = [
    "keycloak",    # Identity and Access Management (accounts)
    "org",         # Additional tools for the Organization in Go
    "opentdf",     # E2EE Solution
    "AICouncil",   # Agentic Solution in Python
    "nginx",       # Reverse proxy for Reactor and SSH
    #"ollama",      # LLM Host
    #"sglang",      # LLM Host
    #"bluesky",     # Social Media - PDS for Bluesky
    "synapse",     # Social Media - Matrix Server
    "element",     # Social Media - Matrix Client
    "webapp",      # Social Media - Our Client
    "webapp_build",# Social Media - Our Client Production Build
]

distinguisher = ""  # If you are running multiple deployments on the same machine, you can distinguish them here
KEYCLOAK_PORT = ""  # if applicable
KEYCLOAK_INTERNAL_URL = "keycloak." + BACKEND_LOCATION
SYNAPSE_CLIENT_SECRET = "changeme"

# OAUTH Config
# Google OAuth Config
GOOGLE_CLIENT_SECRET = "<YOUR SECRET HERE>"
GOOGLE_CLIENT_ID = "<YOUR GOOGLE OAUTH CLIENT ID>"
GOOGLE_SCOPES = "openid profile email"
GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

# GitHub OAuth Config
GITHUB_CLIENT_SECRET = "<YOUR SECRET HERE>"
GITHUB_CLIENT_ID = "<YOUR GITHUB OAUTH CLIENT ID>"
GITHUB_SCOPES = "openid profile email"
GITHUB_AUTH_ENDPOINT = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_ENDPOINT = "https://github.com/login/oauth/access_token"

PEM_FILE = "<YOUR SECRET HERE>"
SERVER_USER = "<YOUR SECRET HERE>"
SERVER_HOST = "<YOUR SECRET HERE>"
REMOTE_FOLDER = "<YOUR SECRET HERE>"
ZIP_FILE = "<YOUR SECRET HERE>"
LOCAL_DESTINATION = "<YOUR SECRET HERE>"
EXTRACT_FOLDER = "<YOUR SECRET HERE>"  # Name of the folder after extraction

BLUESKY_HANDLE = "<YOUR SECRET HERE>"
BLUESKY_PASSWORD = "<YOUR SECRET HERE>"

import os
import shutil
import sys
import util
import time
import os
import sys
import utils_docker
import json

here = os.path.abspath(os.path.dirname(__file__))

# create env.py file if this is the first run
util.initializeFiles()

print("Reading env.py")
import env

print("Applying env var substitutions in hard-coded .template files")
util.substitutions(here, env)
util.writeViteEnv(vars(env))

if not os.path.isdir(env.keys_dir):
    if env.USER_WEBSITE == "localhost":
        os.system("cd certs && ./init-temp-keys.cmd")
    else:
        utils_docker.generateProdKeys(env)

# Convert env.py to a dictionary
config = vars(env)
# make sure the network is up
utils_docker.ensure_network(env.NETWORK_NAME)

# create the keycloak keys if they dont exist
if not os.path.isdir("certs/keys"):
    os.system("cd certs && ./init-temp-keys.sh")

# --- KEYCLOAK ---
if "keycloak" in env.SERVICES_TO_RUN:
    utils_docker.run_container(env.keycloakdb)
    utils_docker.wait_for_db(network=env.NETWORK_NAME, db_url="keycloakdb:5432")
    utils_docker.run_container(env.keycloak)

# --- WEB APP ---
# theoretically has no dependencies
if "webapp" in env.SERVICES_TO_RUN:
    utils_docker.run_container(env.webapp)
if "webapp_build" in env.SERVICES_TO_RUN:
    utils_docker.run_container(env.webapp_build)

# --- NGINX ---
if "nginx" in env.SERVICES_TO_RUN:
    if not os.path.isfile("certs/ca.crt"):
        if env.USER_WEBSITE == "localhost":
            utils_docker.generateDevKeys(outdir=env.certs_dir)
        else:
            pass
            #utils_docker.generateProdKeys(outdir=env.certs_dir, website=env.USER_WEBSITE)
    utils_docker.run_container(env.nginx)

# --- OPENTDF ---
if "opentdf" in env.SERVICES_TO_RUN:
    utils_docker.run_container(env.opentdfdb)
    utils_docker.wait_for_db(network=env.NETWORK_NAME, db_url="opentdfdb:5432")
    utils_docker.wait_for_url(env.KEYCLOAK_INTERNAL_AUTH_URL, network=env.NETWORK_NAME)
    utils_docker.run_container(env.opentdf)

# --- ORG ---
if "org" in env.SERVICES_TO_RUN:
    utils_docker.run_container(env.org)

# --- MATRIX SYNAPSE ---
if "synapse" in env.SERVICES_TO_RUN:
    #docker run -it --rm \
    #--mount type=volume,src=synapse-data,dst=/data \
    #-e SYNAPSE_SERVER_NAME=my.matrix.host \
    #-e SYNAPSE_REPORT_STATS=yes \
    #matrixdotorg/synapse:latest generate

    utils_docker.run_container(env.synapsedb)
    utils_docker.wait_for_db(network=env.NETWORK_NAME, db_url="synapsedb:5432")
    utils_docker.run_container(env.synapse)
if "element" in env.SERVICES_TO_RUN:
    # --- Element web app ---
    utils_docker.run_container(env.element)

# --- Discourse ---
if "discourse" in env.SERVICES_TO_RUN:
    utils_docker.run_container(env.discourse)

# --- OLLAMA !!! ---
if "ollama" in env.SERVICES_TO_RUN:
    utils_docker.run_container(env.ollama)
    utils_docker.pullModels(["llama3.2", "ALIENTELLIGENCE/sigmundfreud", "phi3"],env.NETWORK_NAME)

# --- BLUESKY PDS --- 
if "bluesky" in env.SERVICES_TO_RUN:
    utils_docker.run_container(env.bluesky)
    utils_docker.run_container(env.bluesky_bridge)
    utils_docker.run_container(env.bsky_fyp)

if "irc" in env.SERVICES_TO_RUN:
    utils_docker.run_container(env.irc)
    utils_docker.run_container(env.thelounge)
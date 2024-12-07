import os
import shutil
import sys
import util
import time
import os
import sys
import utils_docker

here = os.path.abspath(os.path.dirname(__file__))

# create env.py file if this is the first run
util.initializeFiles()

print("Reading env.py")
import env

print("Applying env var substitutions in hard-coded .template files")
util.substitutions(here, env)

# Convert env.py to a dictionary
config = vars(env)
# make sure the network is up
utils_docker.ensure_network(env.BRAND_NAME)

# --- WEB APP ---
# theoretically has no dependencies
utils_docker.run_container(env.webapp)

# --- KEYCLOAK ---
utils_docker.run_container(env.keycloakdb)
utils_docker.wait_for_db(network=env.BRAND_NAME, db_url="keycloakdb:5432")
# create the keys if they dont exist
if not os.path.isdir("keycloak/keys"):
    os.system("cd keycloak && ./init-temp-keys.sh")
    os.system("cd keycloak/keys && chmod 666 *")
utils_docker.run_container(env.keycloak)

# --- NGINX ---
if not os.path.isfile("nginx/ca.crt"):
    if env.IS_EC2:
        utils_docker.generateProdKeys(outdir = env.nginx_dir, website=env.USER_WEBSITE)
    else:
        utils_docker.generateDevKeys(outdir = env.nginx_dir)
utils_docker.run_container(env.nginx)

# --- OPENTDF ---
utils_docker.run_container(env.opentdfdb)
utils_docker.wait_for_db(network=env.BRAND_NAME, db_url="opentdfdb:5432")
utils_docker.wait_for_url(env.KEYCLOAK_INTERNAL_CHECK_ADDR, network=env.BRAND_NAME)
utils_docker.run_container(env.opentdf)

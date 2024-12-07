import os
import shutil
import sys
import util 
import time
import os
import sys 
import utils_docker

here = os.path.abspath(os.path.dirname(__file__))

container_name = sys.argv[1]

# create env.py file if this is the first run
util.initializeFiles()

print("Reading env.py")
import env
print("Applying env var substitutions in hard-coded .template files")
util.substitutions(here, env)

# Convert env.py to a dictionary
config = vars(env)
dockerConfig = config[container_name]

# make sure the network is up
utils_docker.ensure_network(env.BRAND_NAME)

# start up keycloak
utils_docker.debug_container(dockerConfig)

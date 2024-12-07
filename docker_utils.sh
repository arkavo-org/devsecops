# docker_utils.sh

check_and_start_container() {
  local container_name="$1"

  # Check if the container is running
  if docker ps --format '{{.Names}}' | grep -q "^${container_name}\$"; then
    echo "Container '${container_name}' is already running. Exiting."
    exit 0
  fi

  # Check if the container exists but is not running
  if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}\$"; then
    echo "Container '${container_name}' exists but is stopped. Starting it."
    docker start "${container_name}" && \
      echo "Container '${container_name}' started." || \
      echo "Failed to start container '${container_name}'."
    exit 0
  fi

  # Indicate that the container needs to be created
  echo "Container '${container_name}' does not exist. It needs to be created."
}

wait_on_url() {
  local container_name="$1"
  local db_url="$2"
  local db_user="$3"

  # Check if required arguments are provided
  if [ -z "$container_name" ] || [ -z "$db_url" ] || [ -z "$db_user" ]; then
    echo "Usage: wait_on_url <container_name> <db_url:port> <db_user>"
    return 1
  fi

  # Wait for the specified database to respond
  docker run --rm --network $BRAND_NAME postgres:15-alpine sh -c "
    echo \"Waiting for ${container_name} to respond on ${db_url}...\";
    until pg_isready -h ${db_url%:*} -p ${db_url##*:} -U ${db_user} >/dev/null 2>&1; do
      sleep 5;
      echo \"Still waiting for ${container_name} to accept connections on ${db_url}...\";
    done;
    echo \"${container_name} is accepting connections on ${db_url}!\";
  "
}


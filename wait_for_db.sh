db_url="$1"
db_user="$2"
echo "Using db_url: $db_url"

# Check if required arguments are provided
if [ -z "$db_url" ] || [ -z "$db_user" ]; then
  echo "Usage: wait_on_db_url <db_url> <db_user>"
  exit 1
fi

# Wait for the specified database to respond
docker run --rm --network $BRAND_NAME postgres:15-alpine sh -c "
  echo \"Waiting for the database to respond on ${db_url}...\";
  until pg_isready -h ${db_url%:*} -p ${db_url##*:} -U ${db_user} >/dev/null 2>&1; do
    sleep 2;
    echo \"Still waiting for the database to accept connections on ${db_url}...\";
  done;
  echo \"The database is accepting connections on ${db_url}!\";
"

docker run --rm --net=host \
  --name="master" \
  -v "$(pwd)":/app \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w /app \
  -e GITHUB_ACTIONS=$GITHUB_ACTIONS \
  -e CURRENT_DIR=$(pwd) \
  -v /usr/bin/docker:/usr/bin/docker \
  python:3.12 bash -c "pip install docker && python3 run.py"

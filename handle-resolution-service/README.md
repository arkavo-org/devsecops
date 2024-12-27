# Handle Resolution Service

ATProto handle resolution service built with AWS SAM.

## Prerequisites

- Python 3.12
```shell
brew install python@3.12
```
- AWS SAM CLI
- AWS CLI configured with appropriate credentials

## Development Setup

1. Create and activate a virtual environment:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

2. Install development dependencies:
```bash
# Update pip
python -m pip install --upgrade pip

# Install development requirements
pip install aws-sam-cli awscli
```

## Project Structure

```
handle-resolution-service/
├── template.yaml          # SAM template
├── requirements.txt       # Python dependencies
├── build.sh              # Build script
├── src/                  # Lambda function code
└── dependencies/         # Lambda layer dependencies
```

## Building and Deployment

1. Build the project:
```bash
chmod +x build.sh
./build.sh
```

2. Deploy (first time):
```bash
sam deploy --guided
```

3. Subsequent deployments:
```bash
sam build && sam deploy --no-confirm-changeset
```

## Local Development

1. Install local dependencies in your virtual environment:
```bash
pip install -r requirements.txt
```

2. Colima
```shell
colima start
export DOCKER_HOST="unix://${HOME}/.colima/docker.sock"
````

3. Redis
```shell
docker run -d --name redis -p 6379:6379 redis
```

4. DynamoDB
```shell
docker run -d --name dynamodb -p 8000:8000 amazon/dynamodb-local
```

```shell
# Create the table
aws dynamodb create-table \
    --table-name dev-handles \
    --attribute-definitions AttributeName=handle,AttributeType=S \
    --key-schema AttributeName=handle,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --endpoint-url http://localhost:8000

# Add test data
aws dynamodb put-item \
    --table-name dev-handles \
    --item '{"handle": {"S": "test.arkavo.social"}, "did": {"S": "did:plc:testuser123"}}' \
    --endpoint-url http://localhost:8000
```


3. Local testing:
```bash
# Test GET request
sam local invoke HandleCheckFunction --env-vars env.json -e events/get-request.json

# Test HEAD request
sam local invoke HandleCheckFunction --env-vars env.json -e events/head-request.json

# Start local API
sam local start-api
```

4. Test the API:
```bash
# Test GET endpoint
curl "http://localhost:3000/xrpc/com.atproto.identity.resolveHandle?handle=test.arkavo.social"

# Test HEAD endpoint
curl -I "http://localhost:3000/xrpc/com.atproto.identity.resolveHandle?handle=test.arkavo.social"
```

## API Endpoints

The service provides two endpoints:

### GET /xrpc/com.atproto.identity.resolveHandle
Resolves a handle to a DID.

Query Parameters:
- `handle`: The handle to resolve (e.g., `username.arkavo.social`)

### HEAD /xrpc/com.atproto.identity.resolveHandle
Quick check for handle validity and existence.

Query Parameters:
- `handle`: The handle to check (e.g., `username.arkavo.social`)

## Contributing

1. Create a new branch for your feature
2. Make changes in your branch
3. Test locally using SAM CLI
4. Submit a pull request

## Common Issues

### Virtual Environment Issues

If you see "command not found" after activating the virtual environment:
```bash
# Deactivate and remove the current venv
deactivate
rm -rf .venv

# Create a new venv with --clear flag
python3.12 -m venv .venv --clear

# Reactivate and reinstall dependencies
source .venv/bin/activate
pip install -r requirements.txt
```
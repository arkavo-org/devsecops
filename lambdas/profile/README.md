# Arkavo Profile Lambda

ATProtocol-compliant Lambda function for retrieving user profile information. This service provides a scalable endpoint for accessing both standard ATProtocol profile fields and Arkavo-specific extensions.

## Overview

The Profile Lambda function serves as an endpoint for retrieving user profile information through the ATProtocol specification. It interfaces with DynamoDB to store and retrieve profile data, handling both standard ATProtocol fields and Arkavo-specific extensions.

### Endpoint Details

- **URL**: `https://xrpc.arkavo.net/xrpc/app.arkavo.actor.getProfile`
- **Method**: GET
- **Query Parameters**:
  - `actor` (required): String representing either the DID or handle of the user

## Getting Started

### Prerequisites

- Python 3.9+
- AWS CLI configured with appropriate credentials
- AWS SAM CLI (for local development)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/arkavo/devsecops.git
cd lambdas/profile
```

2. Create a virtual environment and install dependencies:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Set up local environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Local Development

Run the function locally using SAM CLI:

```bash
sam local invoke -e events/test_event.json
```

Or start a local API endpoint:

```bash
sam local start-api
```

## Deployment

### Using AWS SAM

1. Build the application:
```bash
sam build
```

2. Deploy to AWS:
```bash
sam deploy --guided
```

### Manual Deployment

1. Package the Lambda:
```bash
zip -r function.zip . -x "*.git*" "*.env*" "*.venv*"
```

2. Deploy using AWS CLI:
```bash
aws lambda update-function-code --function-name profile-lambda --zip-file fileb://function.zip
```

## Configuration

### Environment Variables

- `PROFILES_TABLE_NAME`: DynamoDB table name for storing profiles

### DynamoDB Schema

The DynamoDB table requires the following schema:

```json
{
  "actor": "string" (Primary Key),
  "handle": "string",
  "did": "string",
  "displayName": "string",
  "avatarUrl": "string" (optional),
  "description": "string" (optional),
  "profileType": "string",
  "isVerified": "boolean",
  "followersCount": "number",
  "followingCount": "number",
  "creationDate": "string",
  "publicID": "string"
}
```

## API Response Format

### Successful Response

```json
{
  "standard": {
    "handle": "string",
    "did": "string",
    "displayName": "string",
    "avatarUrl": "string",
    "description": "string"
  },
  "extended": {
    "profileType": "string",
    "isVerified": boolean,
    "followersCount": number,
    "followingCount": number,
    "creationDate": "string",
    "publicID": "string"
  }
}
```

### Error Response

```json
{
  "error": "Error message"
}
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Monitoring and Logging

The function uses AWS Lambda Powertools for structured logging and tracing. Logs are available in CloudWatch Logs, and traces can be viewed in AWS X-Ray.

### Log Groups

- `/aws/lambda/profile-lambda`: Main function logs
- `/aws/lambda/profile-lambda-dev`: Development environment logs

### Metrics

Key metrics are available in CloudWatch Metrics under the custom namespace `Arkavo/ProfileLambda`:

- `GetProfileLatency`: Response time for profile retrieval
- `ProfileNotFound`: Count of 404 errors
- `DatabaseErrors`: Count of DynamoDB-related errors

## Related Documentation

- [ATProtocol Specification](https://atproto.com/specs/atp)
- [AWS Lambda Powertools](https://awslabs.github.io/aws-lambda-powertools-python/)
- 
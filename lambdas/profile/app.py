import json
import os
from typing import Dict, Any, Optional

import boto3
from aws_lambda_powertools import Logger, Metrics
from aws_lambda_powertools.event_handler import APIGatewayRestResolver
from aws_lambda_powertools.utilities.typing import LambdaContext
from botocore.exceptions import ClientError

# Initialize utilities
logger = Logger()
metrics = Metrics()
app = APIGatewayRestResolver()

# Constants
DYNAMO_TABLE_NAME = os.environ["PROFILES_TABLE_NAME"]


class ProfileError(Exception):
    """Custom exception for profile-related errors"""

    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


def get_profile_from_dynamo(actor: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve profile from DynamoDB
    """
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(DYNAMO_TABLE_NAME)

    try:
        # First try as handle (primary key)
        response = table.get_item(Key={"handle": actor})
        if "Item" in response:
            return response["Item"]

        # If not found, scan for DID or publicID
        response = table.scan(
            FilterExpression="did = :did OR publicID = :pid",
            ExpressionAttributeValues={
                ":did": actor,
                ":pid": actor
            }
        )
        items = response.get("Items", [])
        return items[0] if items else None
    except ClientError as e:
        logger.error(f"DynamoDB error: {str(e)}")
        raise ProfileError("Database error", 500)


def validate_profile_data(profile: Dict[str, Any]) -> None:
    """
    Validate profile data structure and required fields
    """
    required_fields = {
        "handle", "did", "profileName", "creationDate", "publicID"
    }

    missing_fields = required_fields - set(profile.keys())
    if missing_fields:
        raise ProfileError(f"Missing required fields: {', '.join(missing_fields)}")


def format_profile_response(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format profile data according to ATProtocol specification
    """
    return {
        "standard": {
            "handle": profile["handle"],
            "did": profile["did"],
            "displayName": profile["profileName"],
            "avatarUrl": profile.get("avatarUrl"),
            "description": profile.get("description")
        },
        "extended": {
            "creationDate": profile["creationDate"],
            "publicID": profile["publicID"]
        }
    }


@app.get("/xrpc/app.arkavo.actor.getProfile")
def get_profile() -> Dict[str, Any]:
    """
    Main handler for profile retrieval
    """
    try:
        # Get and validate query parameters
        actor = app.current_event.get_query_string_value(name="actor")
        if not actor:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'actor' parameter"})
            }

        # Get profile data
        profile = get_profile_from_dynamo(actor)
        if not profile:
            return {
                "statusCode": 404,
                "body": json.dumps({"error": "Profile not found"})
            }

        # Validate profile data
        validate_profile_data(profile)

        # Record metric for successful lookup
        metrics.add_metric(name="ProfileLookupSuccess", unit="Count", value=1)

        # Format and return response
        formatted_response = format_profile_response(profile)
        return {
            "statusCode": 200,
            "body": json.dumps(formatted_response)
        }

    except ProfileError as error:
        logger.error(f"Profile error: {error.message}")
        metrics.add_metric(name="ProfileLookupError", unit="Count", value=1)
        return {
            "statusCode": error.status_code,
            "body": json.dumps({"error": error.message})
        }
    except Exception as e:
        logger.exception("Unhandled exception")
        metrics.add_metric(name="UnhandledError", unit="Count", value=1)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error"})
        }


@logger.inject_lambda_context
@metrics.log_metrics
def lambda_handler(event: Dict[str, Any], context: LambdaContext) -> Dict[str, Any]:
    """
    Lambda handler entry point
    """
    return app.resolve(event, context)


if __name__ == "__main__":
    # For local testing
    test_event = {
        "queryStringParameters": {"actor": "test_user"}
    }
    print(lambda_handler(test_event, None))

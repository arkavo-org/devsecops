import re
import json
import os
import redis
import boto3
from typing import Optional, Dict, Any, Tuple


def get_redis_client():
    """
    Get Redis client with appropriate configuration for environment
    """
    is_local = os.environ.get('AWS_SAM_LOCAL') == 'true'

    if is_local:
        # Local development settings
        return redis.Redis(
            host='host.docker.internal',  # Special Docker DNS name to reach host machine
            port=6379,
            decode_responses=True,
            socket_connect_timeout=1,  # Short timeout for development
            retry_on_timeout=False
        )
    else:
        # Production settings
        return redis.Redis(
            host=os.environ['REDIS_ENDPOINT'],
            port=6379,
            ssl=True,
            decode_responses=True
        )


def get_dynamodb_table():
    """
    Get DynamoDB table with appropriate configuration for environment
    """
    is_local = os.environ.get('AWS_SAM_LOCAL') == 'true'

    if is_local:
        dynamodb = boto3.resource('dynamodb', endpoint_url='http://host.docker.internal:8000')
        table_name = os.environ.get('DYNAMODB_TABLE', 'dev-handles')
    else:
        dynamodb = boto3.resource('dynamodb')
        table_name = os.environ['DYNAMODB_TABLE']

    return dynamodb.Table(table_name)


# Initialize clients
try:
    redis_client = get_redis_client()
    table = get_dynamodb_table()
except Exception as e:
    print(f"Warning: Failed to initialize clients: {str(e)}")
    redis_client = None
    table = None


def validate_handle(handle: str) -> bool:
    """
    Validate handle format according to ATProto specs.
    Must be a valid DNS name.
    """
    if not handle or not isinstance(handle, str):
        return False

    # Split into parts (e.g., "user.arkavo.social" -> ["user", "arkavo", "social"])
    parts = handle.split('.')
    if len(parts) < 2:  # Must have at least one subdomain
        return False

    # Validate each DNS label
    for part in parts:
        # Length check (1-63 characters)
        if not 1 <= len(part) <= 63:
            return False

        # Must start with alphanumeric, end with alphanumeric,
        # and contain only alphanumeric or hyphens
        if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$', part):
            return False

    return True


def get_handle_data(handle: str) -> Tuple[Optional[Dict[str, str]], bool]:
    """
    Retrieve handle data from cache or database.
    Returns tuple of (data, is_cached).
    """
    if not redis_client or not table:
        # Return mock data for local development if services aren't available
        return {
            'did': 'did:plc:mock12345',
            'handle': handle
        }, False

    # Check cache first
    try:
        cache_key = f'handle:{handle.lower()}'
        cached_result = redis_client.get(cache_key)

        if cached_result:
            try:
                return json.loads(cached_result), True
            except json.JSONDecodeError:
                # Invalid cache entry, will fetch from DB
                redis_client.delete(cache_key)
    except redis.RedisError as e:
        print(f"Redis error: {str(e)}")
        # Continue without cache

    # Query DynamoDB if not in cache or cache was invalid
    try:
        response = table.get_item(
            Key={'handle': handle.lower()},
            ConsistentRead=True,
            ProjectionExpression='handle, did'
        )
    except Exception as e:
        print(f"DynamoDB error: {str(e)}")
        return None, False

    item = response.get('Item')
    if not item:
        return None, False

    result = {
        'did': item['did'],
        'handle': handle
    }

    # Try to cache the result (1 hour TTL)
    if redis_client:
        try:
            redis_client.setex(
                cache_key,
                3600,  # 1 hour
                json.dumps(result)
            )
        except redis.RedisError as e:
            print(f"Redis caching error: {str(e)}")

    return result, False


def create_response(
        status_code: int,
        body: Optional[Dict[str, Any]] = None,
        cached: bool = False,
        method: str = 'GET'
) -> Dict[str, Any]:
    """
    Create API Gateway response with proper headers.
    """
    headers = {}

    if method == 'HEAD':
        # Minimal headers for HEAD requests
        headers['Access-Control-Allow-Origin'] = '*'
    else:
        # Full headers for GET requests
        headers.update({
            'Content-Type': 'application/json',
            'X-Cache-Hit': str(cached).lower(),
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, HEAD',
            'Access-Control-Allow-Headers': 'Content-Type'
        })

    response = {
        'statusCode': status_code,
        'headers': headers
    }

    if body is not None and method != 'HEAD':
        response['body'] = json.dumps(body)

    return response


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for handle resolution.
    Supports both GET and HEAD methods.
    """
    # Extract HTTP method and handle
    method = event.get('httpMethod', 'GET')
    handle = event.get('queryStringParameters', {}).get('handle')

    # Validate handle format
    if not handle or not validate_handle(handle):
        return create_response(
            400,
            {'error': 'Invalid handle format'} if method == 'GET' else None,
            method=method
        )

    # For HEAD requests, we only validate the handle
    if method == 'HEAD':
        return create_response(200, method=method)

    # For GET requests, fetch and return the data
    result, cached = get_handle_data(handle)

    if not result:
        return create_response(
            404,
            {'error': 'Handle not found'},
            method=method
        )

    return create_response(200, result, cached, method=method)

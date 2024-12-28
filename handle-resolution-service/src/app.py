import re
import json
import os
import redis
import boto3
import logging
from typing import Optional, Dict, Any, Tuple
from botocore.config import Config

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configure boto3 with retries
boto3_config = Config(
    retries=dict(
        max_attempts=3
    )
)


def get_redis_client():
    """
    Get Redis client with appropriate configuration for environment.
    Uses REDIS_ENDPOINT environment variable.
    """
    is_local = os.environ.get('AWS_SAM_LOCAL') == 'true'

    logger.info(f"Redis initialization - is_local: {is_local}")

    if is_local:
        # Local development settings
        return redis.Redis(
            host='host.docker.internal',
            port=6379,
            decode_responses=True,
            socket_connect_timeout=1,
            retry_on_timeout=False
        )
    else:
        # Production settings
        redis_host = os.environ.get('REDIS_ENDPOINT')
        if not redis_host:
            raise ValueError("REDIS_ENDPOINT environment variable not set")

        logger.info(f"Connecting to Redis at: {redis_host}")

        connection_kwargs = {
            'host': redis_host,
            'port': 6379,
            'decode_responses': True,
            'socket_timeout': 3,
            'socket_connect_timeout': 3,
            'retry_on_timeout': True,
            'max_connections': 10,
            'connection_class': redis.SSLConnection,
        }

        try:
            pool = redis.ConnectionPool(**connection_kwargs)
            return redis.Redis(connection_pool=pool)
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise


def get_dynamodb_table():
    """
    Get DynamoDB table with appropriate configuration for environment
    """
    is_local = os.environ.get('AWS_SAM_LOCAL') == 'true'

    if is_local:
        dynamodb = boto3.resource('dynamodb',
                                  endpoint_url='http://host.docker.internal:8000',
                                  config=boto3_config)
        table_name = os.environ.get('DYNAMODB_TABLE', 'dev-handles')
    else:
        dynamodb = boto3.resource('dynamodb', config=boto3_config)
        table_name = os.environ['DYNAMODB_TABLE']

    return dynamodb.Table(table_name)


# Initialize clients
try:
    redis_client = get_redis_client()
    table = get_dynamodb_table()
    logger.info("Successfully initialized clients")
except Exception as e:
    logger.error(f"Failed to initialize clients: {str(e)}")
    raise


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
    cache_key = f'handle:{handle.lower()}'
    logger.info(f"Starting to fetch data for handle: {handle} (Cache Key: {cache_key})")

    # Step 1: Check Redis cache
    try:
        logger.info(f"Checking Redis cache for key: {cache_key}")
        cached_result = redis_client.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for key: {cache_key}")
            try:
                # Attempt to parse the cached result
                parsed_result = json.loads(cached_result)
                logger.info(f"Successfully parsed cached result: {parsed_result}")
                # Only return the 'did' field
                return {'did': parsed_result['did']}, True
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse cached result for key: {cache_key}. Error: {str(e)}")
                try:
                    # Delete the invalid cache entry
                    logger.info(f"Deleting invalid cache entry for key: {cache_key}")
                    redis_client.delete(cache_key)
                except redis.RedisError as e:
                    logger.error(f"Failed to delete invalid cache entry for key: {cache_key}. Error: {str(e)}")
                    pass  # Ignore delete errors
        else:
            logger.info(f"Cache miss for key: {cache_key}")
    except (redis.ConnectionError, redis.TimeoutError) as e:
        logger.error(f"Redis connection/timeout error while fetching key: {cache_key}. Error: {str(e)}")
    except redis.RedisError as e:
        logger.error(f"Redis error while fetching key: {cache_key}. Error: {str(e)}")

    # Step 2: Query DynamoDB if not in cache or cache was invalid
    try:
        logger.info(f"Querying DynamoDB for handle: {handle}")
        response = table.get_item(
            Key={'handle': handle.lower()},
            ConsistentRead=True,
            ProjectionExpression='did'  # Only request the 'did' field
        )
        logger.info(f"DynamoDB response: {response}")
    except Exception as e:
        logger.error(f"DynamoDB error while querying handle: {handle}. Error: {str(e)}")
        raise

    # Step 3: Process DynamoDB response
    item = response.get('Item')
    if not item:
        logger.info(f"Handle not found in DynamoDB: {handle}")
        return None, False

    result = {
        'did': item['did']
    }
    logger.info(f"Successfully retrieved data from DynamoDB: {result}")

    # Step 4: Cache the result in Redis
    try:
        logger.info(f"Caching result in Redis for key: {cache_key} with TTL: 3600 seconds")
        redis_client.setex(
            cache_key,
            3600,  # 1 hour
            json.dumps(result)
        )
        logger.info(f"Successfully cached result for key: {cache_key}")
    except redis.RedisError as e:
        logger.error(f"Redis caching error for key: {cache_key}. Error: {str(e)}")

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
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, HEAD',
        'Access-Control-Allow-Headers': 'Content-Type'
    }

    if method != 'HEAD':
        headers.update({
            'Content-Type': 'application/json',
            'X-Cache-Hit': str(cached).lower()
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
    logger.info(f"Processing request: {json.dumps(event)}")

    # Extract method before try block to ensure it's available in error handling
    method = event.get('httpMethod', 'GET')

    try:
        # Extract handle from query parameters
        query_params = event.get('queryStringParameters', {}) or {}
        handle = query_params.get('handle')

        # Validate handle format
        if not handle or not validate_handle(handle):
            return create_response(
                400,
                {'error': 'Invalid handle format'} if method == 'GET' else None,
                method=method
            )

        # For both GET and HEAD requests, fetch the data
        result, cached = get_handle_data(handle)

        if not result:
            return create_response(
                404,
                {'error': 'Handle not found'} if method == 'GET' else None,
                method=method
            )

        # Return appropriate response based on method
        return create_response(
            200,
            result if method == 'GET' else None,
            cached,
            method=method
        )

    except redis.ConnectionError as e:
        logger.error(f"Redis connection error: {str(e)}")
        return create_response(
            503,
            {'error': 'Service temporarily unavailable'} if method == 'GET' else None,
            method=method
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return create_response(
            500,
            {'error': 'Internal server error'} if method == 'GET' else None,
            method=method
        )

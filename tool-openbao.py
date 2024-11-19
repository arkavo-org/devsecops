import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from main import load_env

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Secret(BaseModel):
    """Model for secret data."""
    path: str
    data: Dict[str, Any]


class CertificateRequest(BaseModel):
    """Model for certificate request data."""
    common_name: str
    ttl: str = "8760h"  # Default 1 year
    alt_names: Optional[List[str]] = None
    ip_sans: Optional[List[str]] = None
    uri_sans: Optional[List[str]] = None
    format: str = "pem"  # Default format


class OpenBaoTool(BaseTool):
    """Base tool for interacting with OpenBao's API."""

    name: str = Field(default="openbao_tool")
    description: str = Field(default="Base tool for OpenBao API interactions")
    api_url: str = Field(..., description="Base URL for OpenBao API")
    token: str = Field(..., description="Authentication token for OpenBao")
    return_direct: bool = Field(default=True)

    def _make_request(
            self,
            method: str,
            endpoint: str,
            data: Optional[Dict[str, Any]] = None,
            params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a request to the OpenBao API."""
        headers = {
            "X-Vault-Token": self.token,
            "Content-Type": "application/json",
            "X-Vault-Request": "true"
        }

        url = f"{self.api_url.rstrip('/')}/{endpoint.lstrip('/')}"

        if method == "LIST":
            # OpenBao uses GET with ?list=true for listing
            method = "GET"
            params = params or {}
            params["list"] = "true"

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params,
                verify=True
            )
            response.raise_for_status()
            if response.text:  # Only parse JSON if there's content
                return response.json()
            return {}  # Return empty dict for no content
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise


class KVTool(OpenBaoTool):
    """Tool for managing KV secrets in OpenBao."""

    name: str = Field(default="kv_tool")
    description: str = Field(default="Tool for managing KV secrets")

    def list_secrets(self, path: str) -> List[str]:
        """List secrets at the specified path."""
        # For KV v2, use metadata for listing
        endpoint = f"v1/secret/metadata/{path.rstrip('/')}"
        response = self._make_request("LIST", endpoint)
        return response.get("data", {}).get("keys", [])

    def read_secret(self, path: str) -> Dict[str, Any]:
        """Read a secret from the specified path."""
        endpoint = f"v1/secret/data/{path}"
        return self._make_request("GET", endpoint)

    def write_secret(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Write a secret to the specified path."""
        endpoint = f"v1/secret/data/{path}"
        wrapped_data = {"data": data}
        return self._make_request("POST", endpoint, data=wrapped_data)

    def delete_secret(self, path: str) -> None:
        """Delete a secret at the specified path."""
        # For KV v2, use metadata endpoint for deletion
        endpoint = f"v1/secret/metadata/{path}"
        self._make_request("DELETE", endpoint)

    def _run(self, query: str) -> str:
        """Execute KV operations based on query."""
        try:
            parts = query.split(":", 2)
            if len(parts) < 2:
                return "Invalid query format. Expected 'action:path[:data]'"

            action, path = parts[0].lower(), parts[1]

            # Handle empty path for root listing
            path = path or "/"

            if action == "list":
                result = self.list_secrets(path)
            elif action == "read":
                result = self.read_secret(path)
            elif action == "write" and len(parts) == 3:
                data = json.loads(parts[2])
                result = self.write_secret(path, data)
            elif action == "delete":
                self.delete_secret(path)
                result = {"status": "deleted"}
            else:
                raise ValueError(f"Invalid action: {action}")

            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"


if __name__ == "__main__":
    load_env()
    # Configuration
    API_URL: str = os.environ.get('OPENBAO_URL')
    TOKEN: str = os.environ.get('OPENBAO_TOKEN')

    try:
        # Create KV tool
        kv_tool = KVTool(api_url=API_URL, token=TOKEN)

        # Example 1: KV Operations
        print("\n=== KV Operations ===")

        # Test connection first
        print(f"{API_URL}/v1/sys/health")
        kv_tool._make_request("GET", "v1/sys/health")
        print("Connection successful!")

        # List secrets in the root path first
        print("\nListing root secrets:")
        list_query = "list:"
        print(kv_tool.run(list_query))

        # Write a secret
        secret_data = {
            "credentials": {
                "username": "admin",
                "password": "secure_password"
            },
            "metadata": {
                "environment": "production",
                "last_updated": datetime.now().isoformat()
            }
        }

        print("\nWriting secret:")
        write_query = f"write:my-app:{json.dumps(secret_data)}"
        print(kv_tool.run(write_query))

        # Read the secret
        print("\nReading secret:")
        read_query = "read:my-app"
        print(kv_tool.run(read_query))

        # List secrets again to see the new one
        print("\nListing secrets after write:")
        print(kv_tool.run(list_query))

        # Clean up
        print("\nDeleting secret:")
        delete_query = "delete:my-app"
        print(kv_tool.run(delete_query))

        print("\nOperations completed successfully")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

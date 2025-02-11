import docker
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Type
from langchain.tools import Tool
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool


class DockerLogAnalysisInput(BaseModel):
    """Input schema for Docker log analysis."""
    container_name: str = Field(..., description="Name or ID of the container")
    time_range_minutes: Optional[int] = Field(default=60, description="Number of minutes of logs to analyze")
    filters: Optional[Dict[str, str]] = Field(default=None, description="Dictionary of filters to apply to logs")
    max_lines: Optional[int] = Field(default=1000, description="Maximum number of log lines to analyze")


class DockerTools:
    """Class containing Docker-related tools for container management and log analysis."""

    def __init__(self):
        self.client = self._initialize_docker_client()

    @staticmethod
    def _initialize_docker_client():
        """Initialize Docker client with support for both Docker Desktop and Colima."""
        socket_paths = [
            os.path.expanduser("~/.colima/default/docker.sock"),  # Colima default
            "/var/run/docker.sock",  # Standard Docker socket
            os.path.expanduser("~/Library/Containers/com.docker.docker/Data/docker.sock")  # Docker Desktop on macOS
        ]

        for socket_path in socket_paths:
            if os.path.exists(socket_path):
                try:
                    client = docker.DockerClient(base_url=f"unix://{socket_path}")
                    client.ping()
                    print(f"Successfully connected to Docker daemon at {socket_path}")
                    return client
                except Exception as e:
                    print(f"Failed to connect to Docker daemon at {socket_path}: {str(e)}")
                    continue

        raise Exception(
            "Could not connect to Docker daemon. Please ensure Docker or Colima is running. "
            "For Colima, run: 'colima start' and try again."
        )

    def list_containers(self, show_all: Any = True) -> str:
        """
        List Docker containers with support for both string and boolean show_all parameter.

        Args:
            show_all: Can be boolean True/False or string 'True'/'False'
        """
        try:
            # Handle both string and boolean input
            if isinstance(show_all, str):
                show_all_bool = show_all.lower() == 'true'
            else:
                show_all_bool = bool(show_all)

            # Try to connect and list containers
            containers = self.client.containers.list(all=show_all_bool)

            if not containers:
                return "SYSTEM STATUS: No Docker containers found. Docker daemon is running but no containers exist."

            result = "CONTAINER ID\tIMAGE\tSTATUS\tNAMES\n"
            for container in containers:
                result += f"{container.short_id}\t{container.image.tags[0] if container.image.tags else 'none'}\t{container.status}\t{container.name}\n"
            return result

        except docker.errors.APIError as e:
            return f"SYSTEM ERROR: Docker daemon not responding. Error: {str(e)}"
        except Exception as e:
            return f"SYSTEM ERROR: {str(e)}"


    def _extract_log_patterns(self, logs: str) -> Dict[str, Any]:
        """Analyze logs for common patterns and anomalies."""
        lines = logs.split('\n')
        analysis = {
            "total_lines": len(lines),
            "error_count": sum(1 for line in lines if 'error' in line.lower()),
            "warning_count": sum(1 for line in lines if 'warn' in line.lower()),
            "patterns": {},
            "timestamps": []
        }

        # Extract timestamps if they exist
        for line in lines:
            try:
                if line and len(line) > 20:
                    timestamp_str = line[:23]
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                    analysis["timestamps"].append(timestamp)
            except (ValueError, IndexError):
                continue

        return analysis

    def analyze_logs(self,
                     container_name: str,
                     time_range_minutes: Optional[int] = 60,
                     filters: Optional[Dict[str, str]] = None,
                     max_lines: Optional[int] = 1000) -> Dict[str, Any]:
        """Analyze logs from a specific container with pattern detection."""
        try:
            container = self.client.containers.get(container_name)

            # Get logs with timestamp
            since = datetime.utcnow() - timedelta(minutes=time_range_minutes)
            logs = container.logs(
                since=since,
                until=datetime.utcnow(),
                timestamps=True,
                tail=max_lines
            ).decode('utf-8')

            # Apply filters if specified
            if filters:
                filtered_logs = []
                for line in logs.split('\n'):
                    if all(value.lower() in line.lower() for value in filters.values()):
                        filtered_logs.append(line)
                logs = '\n'.join(filtered_logs)

            # Analyze logs
            analysis = self._extract_log_patterns(logs)

            # Add container info
            container_info = container.attrs
            analysis["container_info"] = {
                "id": container_info["Id"][:12],
                "name": container_info["Name"],
                "state": container_info["State"]["Status"],
                "created": container_info["Created"]
            }

            return {
                "success": True,
                "analysis": analysis,
                "raw_logs": logs if len(logs) < 1000 else f"{logs[:1000]}... (truncated)"
            }

        except docker.errors.NotFound:
            return {
                "success": False,
                "error": f"Container {container_name} not found"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
        

    def containerRunning(self, container_name):
        """
        Check if a Docker container is running using subprocess.

        :param container_name: Name of the Docker container
        :return: True if the container is running, otherwise False
        """
        # Get the container status
        result = subprocess.run(
            ["docker", "inspect", "--format='{{.State.Status}}'", container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        if result.returncode != 0:
            if "No such object" in result.stderr:
                print(f"Container {container_name} not found")
                return False
            else:
                print(f"Error checking container status: {result.stderr}")
                raise RuntimeError(result.stderr)

        # Extract the status
        status = result.stdout.strip().strip("'")
        if status == "running":
            print(f"Container {container_name} is already running")
            return True

        print(f"Container {container_name} is in status '{status}'. Removing...")
        # Remove the container if it exists but is not running
        remove_result = subprocess.run(
            ["docker", "rm", container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if remove_result.returncode != 0:
            print(f"Error removing container {container_name}: {remove_result.stderr}")
            raise RuntimeError(remove_result.stderr)

        return False

    def create_network(self, networkName):
        """Create Docker network if not exists"""
        if not self.client.networks.get(networkName):
            self.client.networks.create(networkName)
            print(f"Created Network {networkName}")
            return
        print(f"Network {networkName} already exists")
        

    def wait_for_db(self, db_url, db_user, max_attempts=30, delay=2):
        cmd = f"bash ../wait_for_db.sh {db_url} {db_user}"
        print(cmd)
        os.system(cmd)


def create_docker_tools() -> List[Tool]:
    """Create and return a list of Docker-related tools."""
    docker_tools = DockerTools()

    tools = [
        Tool(
            name="docker_ps",
            description="List all running Docker containers. Use parameter 'show_all=True' to include stopped containers.",
            func=docker_tools.list_containers
        ),
        Tool(
            name="docker_log_analysis",
            description="""Analyze logs from a Docker container with pattern detection. Parameters:
            - container_name: Name or ID of the container
            - time_range_minutes: Number of minutes of logs to analyze (default: 60)
            - filters: Dictionary of key-value pairs to filter logs
            - max_lines: Maximum number of log lines to analyze (default: 1000)""",
            func=docker_tools.analyze_logs
        )
    ]

    return tools




# For standalone usage
def main():
    docker_tools = DockerTools()

    # Example usage
    print("\nListing containers:")
    print(docker_tools.list_containers(show_all=True))

    # Example log analysis
    containers = docker_tools.client.containers.list()
    if containers:
        container_name = containers[0].name
        print(f"\nAnalyzing logs for container {container_name}:")
        analysis = docker_tools.analyze_logs(container_name)
        print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
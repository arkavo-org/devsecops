import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Type

import docker
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


class DockerLogAnalysisInput(BaseModel):
    """Input schema for Docker log analysis."""
    container_name: str = Field(..., description="Name or ID of the container")
    time_range_minutes: Optional[int] = Field(default=60, description="Number of minutes of logs to analyze")
    filters: Optional[Dict[str, str]] = Field(default=None, description="Dictionary of filters to apply to logs")
    max_lines: Optional[int] = Field(default=1000, description="Maximum number of log lines to analyze")


class DockerLogAnalysisTool(BaseTool):
    """Tool for analyzing Docker containers logs using LangGraph."""

    name: str = Field(default="docker_log_analysis", description="The name of the tool")
    description: str = Field(default="Analyzes logs from Docker containers with specified filters and time ranges")
    args_schema: Type[BaseModel] = Field(default=DockerLogAnalysisInput)
    _client: docker.DockerClient = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._client = self._initialize_docker_client()

    @staticmethod
    def _initialize_docker_client():
        """Initialize Docker client with support for both Docker Desktop and Colima."""
        # Possible socket paths for Colima and standard Docker
        socket_paths = [
            os.path.expanduser("~/.colima/default/docker.sock"),  # Colima default
            "/var/run/docker.sock",  # Standard Docker socket
            os.path.expanduser("~/Library/Containers/com.docker.docker/Data/docker.sock")  # Docker Desktop on macOS
        ]

        # Try to find a working Docker socket
        for socket_path in socket_paths:
            if os.path.exists(socket_path):
                try:
                    client = docker.DockerClient(base_url=f"unix://{socket_path}")
                    # Test the connection
                    client.ping()
                    print(f"Successfully connected to Docker daemon at {socket_path}")
                    return client
                except Exception as e:
                    print(f"Failed to connect to Docker daemon at {socket_path}: {str(e)}")
                    continue

        # If we get here, no working socket was found
        raise Exception(
            "Could not connect to Docker daemon. Please ensure Docker or Colima is running. "
            "For Colima, run: 'colima start' and try again."
        )

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

    def _run(
            self,
            container_name: str,
            time_range_minutes: Optional[int] = 60,
            filters: Optional[Dict[str, str]] = None,
            max_lines: Optional[int] = 1000,
    ) -> Dict[str, Any]:
        """
        Run the Docker log analysis tool.

        Args:
            container_name: Name or ID of the container
            time_range_minutes: Number of minutes of logs to analyze
            filters: Dictionary of filters to apply to logs
            max_lines: Maximum number of log lines to analyze
        """
        try:
            # Get container
            container = self._client.containers.get(container_name)

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
                    matches_all = True
                    for key, value in filters.items():
                        if value.lower() not in line.lower():
                            matches_all = False
                            break
                    if matches_all:
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


def create_docker_agent():
    """Create a LangGraph agent with Docker log analysis capabilities."""
    docker_tool = DockerLogAnalysisTool()

    def docker_node(state):
        """Node function for Docker log analysis in LangGraph workflow."""
        if not isinstance(state, ToolMessage):
            return state

        try:
            result = docker_tool.run(**state.tool_input)
            return ToolMessage(content=json.dumps(result, indent=2))
        except Exception as e:
            return ToolMessage(content=json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2))

    return docker_node


def list_containers(docker_tool):
    """List all available containers."""
    try:
        containers = docker_tool._client.containers.list(all=True)
        print("\nAvailable containers:")
        print("{:<20} {:<15} {:<15}".format("CONTAINER NAME", "STATUS", "ID"))
        print("-" * 50)
        for container in containers:
            print("{:<20} {:<15} {:<15}".format(
                container.name,
                container.status,
                container.short_id
            ))
    except Exception as e:
        print(f"Error listing containers: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Docker Log Analysis Tool')
    parser.add_argument('--container', '-c', help='Container name or ID')
    parser.add_argument('--time', '-t', type=int, default=60,
                        help='Time range in minutes (default: 60)')
    parser.add_argument('--max-lines', '-m', type=int, default=1000,
                        help='Maximum number of log lines (default: 1000)')
    parser.add_argument('--filter', '-f', action='append',
                        help='Add filters in key=value format (can be used multiple times)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all available containers')

    args = parser.parse_args()

    try:
        # Create tool instance
        docker_tool = DockerLogAnalysisTool()

        # List containers if requested
        if args.list:
            list_containers(docker_tool)
            return

        if not args.container:
            # print("Please specify a container name/ID or use --list to see available containers")
            # sys.exit(1)
            list_containers(docker_tool)
            return

        # Parse filters
        filters = {}
        if args.filter:
            for f in args.filter:
                try:
                    key, value = f.split('=')
                    filters[key] = value
                except ValueError:
                    print(f"Invalid filter format: {f}. Use key=value format")
                    sys.exit(1)

        # Run analysis
        result = docker_tool.run(
            container_name=args.container,
            time_range_minutes=args.time,
            filters=filters,
            max_lines=args.max_lines
        )

        # Pretty print results
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Ensure Colima is running with: 'colima status'")
        print("2. If not running, start it with: 'colima start'")
        print("3. Check Docker connection with: 'docker ps'")
        sys.exit(1)


if __name__ == "__main__":
    main()
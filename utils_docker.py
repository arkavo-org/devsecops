import docker
import json
import subprocess
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Type
here = os.path.abspath(os.path.dirname(__file__))

DOCKER_CLIENT = docker.from_env()


def list_containers(show_all: bool = False) -> str:
    """List Docker containers."""
    try:
        containers = DOCKER_CLIENT.containers.list(all=show_all)

        if not containers:
            return "No containers found"

        result = "CONTAINER ID\tIMAGE\tSTATUS\tNAMES\n"
        for container in containers:
            result += f"{container.short_id}\t{container.image.tags[0] if container.image.tags else 'none'}\t{container.status}\t{container.name}\n"

        return result

    except Exception as e:
        return f"Error listing containers: {str(e)}"


def _extract_log_patterns(logs: str) -> Dict[str, Any]:
    """Analyze logs for common patterns and anomalies."""
    lines = logs.split("\n")
    analysis = {
        "total_lines": len(lines),
        "error_count": sum(1 for line in lines if "error" in line.lower()),
        "warning_count": sum(1 for line in lines if "warn" in line.lower()),
        "patterns": {},
        "timestamps": [],
    }

    # Extract timestamps if they exist
    for line in lines:
        try:
            if line and len(line) > 20:
                timestamp_str = line[:23]
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                analysis["timestamps"].append(timestamp)
        except (ValueError, IndexError):
            continue

    return analysis


def analyze_logs(
    self,
    container_name: str,
    time_range_minutes: Optional[int] = 60,
    filters: Optional[Dict[str, str]] = None,
    max_lines: Optional[int] = 1000,
) -> Dict[str, Any]:
    """Analyze logs from a specific container with pattern detection."""
    try:
        container = DOCKER_CLIENT.containers.get(container_name)

        # Get logs with timestamp
        since = datetime.utcnow() - timedelta(minutes=time_range_minutes)
        logs = container.logs(
            since=since, until=datetime.utcnow(), timestamps=True, tail=max_lines
        ).decode("utf-8")

        # Apply filters if specified
        if filters:
            filtered_logs = []
            for line in logs.split("\n"):
                if all(value.lower() in line.lower() for value in filters.values()):
                    filtered_logs.append(line)
            logs = "\n".join(filtered_logs)

        # Analyze logs
        analysis = self._extract_log_patterns(logs)

        # Add container info
        container_info = container.attrs
        analysis["container_info"] = {
            "id": container_info["Id"][:12],
            "name": container_info["Name"],
            "state": container_info["State"]["Status"],
            "created": container_info["Created"],
        }

        return {
            "success": True,
            "analysis": analysis,
            "raw_logs": logs if len(logs) < 1000 else f"{logs[:1000]}... (truncated)",
        }

    except docker.errors.NotFound:
        return {"success": False, "error": f"Container {container_name} not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


from docker.errors import NotFound, APIError


def create_network(networkName):
    """Create Docker network if not exists"""
    try:
        DOCKER_CLIENT.networks.get(networkName)
        print(f"Network {networkName} already exists")
        return
    except:
        DOCKER_CLIENT.networks.create(networkName)
        print(f"Created Network {networkName}")
        return


def ensure_network(network_name):
    """Ensure the Docker network exists."""
    try:
        DOCKER_CLIENT.networks.get(network_name)
        print(f"Network {network_name} already exists.")
    except NotFound:
        DOCKER_CLIENT.networks.create(network_name)
        print(f"Network {network_name} created.")


def debug_container(config):
    print(f'\033[4;32mDebugging container {config["name"]}\033[0m')
    container_name = config["name"]

    # Get the container if it exists
    try:
        container = DOCKER_CLIENT.containers.get(container_name)
        print(f"Container {container_name} is in status '{container.status}'")

        if container.status == "running":
            print(f"Container {container_name} is already running")
            return True

        if container.status == "restarting":
            print("Stopping container")
            container.stop()

        # Remove the container if it exists but is not running
        print("Removing container")
        container.remove()
    except Exception as e:
        print(f"Container {container_name} not found or already removed")

    # Modify the configuration to use auto-remove and run in the foreground
    # config["auto_remove"] = True  # Enables --rm equivalent
    config["restart_policy"] = None  # Ensure no restart policy is set
    config["detach"] = False  # Run the container in daemon mode to get container object
    config["tty"] = True  # Allocate a pseudo-TTY for interactive logs
    config["remove"] = False   # equivalent to --rm

    # Now run it
    print("Starting container with debug configuration...")
    DOCKER_CLIENT.containers.run(**config)


def run_container(config):
    print(f'\033[4;32mRunning container {config["name"]}\033[0m')
    container_name = config["name"]
    # Get the container
    try:
        container = DOCKER_CLIENT.containers.get(container_name)
        # Check the container status
        print(f"Container {container_name} is in status '{container.status}'")
        if container.status == "running":
            print(f"Container {container_name} is already running")
            return True
        if container.status == "restarting":
            print("Stopping container")
            container.stop()

        print("Removing")
        container.remove()
        print("Running container!")
    except:
        print(f"No container is running with name {container_name}")
    # Now run it
    print(f"Starting {container_name}")
    DOCKER_CLIENT.containers.run(**config)


def wait_for_db(network, db_url, db_user="postgres", max_attempts=30, delay=2):
    print(f"Using db_url: {db_url}")
    print(f"Waiting for the database to respond on {db_url}...")
    host, port = db_url.split(":")

    while True:
        try:
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    network,
                    "postgres:15-alpine",
                    "sh",
                    "-c",
                    f"pg_isready -h {host} -p {port} -U {db_user} >/dev/null 2>&1",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(f"The database is accepting connections on {db_url}!")
            break
        except subprocess.CalledProcessError:
            print(
                f"Still waiting for the database to accept connections on {db_url}..."
            )
            time.sleep(2)


def wait_for_url(url, network):
    # Create and start the container
    run_container(
        dict(
            image="curlimages/curl:latest",  # Use the curl-specific image
            name="url_test",
            network=network,
            environment={"TEST_URL": url},
            command=[
                "sh",
                "-c",
                """
            while ! curl $TEST_URL; do
                echo 'Waiting for Keycloak at' $TEST_URL
                sleep 2
            done
            echo 'Keycloak is up!'
            """,
            ],
            detach=False,
            remove=True,  # Automatically clean up the container after it stops
        )
    )
    
def generateDevKeys(outdir):
    print("Generating Development Keys with SAN for localhost and nginx")

    openssl_config = (
        "[req]\n"
        "default_bits = 2048\n"
        "prompt = no\n"
        "default_md = sha256\n"
        "distinguished_name = dn\n"
        "req_extensions = req_ext\n"
        "\n"
        "[dn]\n"
        "CN = nginx\n"
        "\n"
        "[req_ext]\n"
        "subjectAltName = @alt_names\n"
        "\n"
        "[alt_names]\n"
        "DNS.1 = nginx\n"
        "DNS.2 = localhost\n"
        "IP.1 = 127.0.0.1\n"
    )

    command = (
        "sh -c \""
        "ls /certs && "
        # Install OpenSSL
        "apk add --no-cache openssl && "
        # Write OpenSSL config to a temporary file
        "echo '" + openssl_config.replace("'", "'\\''") + "' > /tmp/openssl.cnf && "
        # Create CA key
        "openssl genrsa -out /certs/ca.key 2048 && "
        # Create CA certificate (self-signed)
        "openssl req -x509 -new -nodes -key /certs/ca.key -sha256 -days 3650 "
        "-subj '/C=US/ST=CA/L=Local/O=MyOrg/CN=MyCA' -out /certs/ca.crt && "
        # Create server key
        "openssl genrsa -out /certs/privkey.pem 2048 && "
        # Create CSR for server certificate using the SAN config
        "openssl req -new -key /certs/privkey.pem -config /tmp/openssl.cnf -out /tmp/server.csr && "
        # Sign the server CSR with the CA key and certificate
        "openssl x509 -req -in /tmp/server.csr -CA /certs/ca.crt -CAkey /certs/ca.key -CAcreateserial "
        "-days 365 -sha256 -extfile /tmp/openssl.cnf -extensions req_ext -out /certs/server.crt && "
        # Combine server cert and CA cert into a full chain
        "cat /certs/server.crt /certs/ca.crt > /certs/fullchain.pem && "
        # Set permissions
        "chmod 644 /certs/privkey.pem /certs/server.crt /certs/fullchain.pem /certs/ca.crt && "
        # Combine it with the standard trust store
        "cat /certs/ca-certificates.crt /certs/ca.crt /keycloak/keys/keycloak-ca.pem /certs/server.crt > /certs/all-ca-certificates.crt\""

    )

    # Run the container to generate the certificate
    try:
        DOCKER_CLIENT.containers.run(
            image='alpine:latest',
            name='cert_gen',
            command=command,
            volumes={
                outdir: {
                    'bind': '/certs/',
                    'mode': 'rw'
                },
                os.path.join(here, "keycloak"): {
                    'bind': '/keycloak/',
                    'mode': 'rw'
                }
            },
            remove=False,
            tty=True
        )
        print("Certificates generated successfully and stored in:", outdir)
    except Exception as e:
        print(f"Error generating certificates: {e}")


def generateProdKeys(outdir, website):
    DOCKER_CLIENT.containers.run(
        image='certbot/certbot',
        command=[
            'certonly',
            '--manual',
            '--preferred-challenges', 'dns',
            '-d', f"{website}",
            '-d', f"*.{website}"
        ],
        volumes={
            outdir: {
                'bind': '/etc/letsencrypt',
                'mode': 'rw'
            }
        },
        remove=False,
        tty=True,
        stdin_open=True
    )

if __name__ == "__main__":
    generateDevKeys(os.path.join(here, "nginx"))
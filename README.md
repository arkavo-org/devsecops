# DevSecOps
DevSecOps LangGraph

## Setup

### .env

```dotenv
ANTHROPIC_API_KEY=abc
GITLAB_URL=http://gitlab.localhost
GITLAB_REPOSITORY=def
```

### LangGraph

```shell
pip install langgraph langsmith langchain_anthropic
```

### Tools

```shell
pip install langchain_community duckduckgo-search langchain-ollama python-gitlab docker
```

### Patch

```shell
pip install urllib3==1.26.5
pip install langgraph==0.2.50
```

# GitLab CE Docker Setup

This guide helps you set up GitLab Community Edition using Docker Compose on Colima.

## Prerequisites

- Colima installed and running
- Docker and Docker Compose installed
- At least 4GB of RAM allocated to Colima
- At least 50GB of disk space

## Setup Steps

1. Start Colima with sufficient resources:
```bash
colima start --cpu 4 --memory 8 --disk 50
```

2. Create a new directory for GitLab:
```bash
mkdir gitlab-ce
cd gitlab-ce
```

3. Add GitLab hostname to your hosts file:
```bash
sudo echo "127.0.0.1 gitlab.localhost" >> /etc/hosts
```

4. Start GitLab:
```bash
docker-compose up
```

5. `root` password:
```bash
docker-compose exec gitlab grep 'Password:' /etc/gitlab/initial_root_password
```

## First-time Access

1. Wait for GitLab to start (this may take a few minutes)
2. Access GitLab at `http://gitlab.localhost`
3. The first time you visit, you'll be asked to set a password for the root user
4. Default username is `root`

## Important Notes

- Initial startup may take 5-10 minutes
- The first password you set will be for the root user
- SSH is available on port 2224
- HTTP is available on port 80
- HTTPS is available on port 443

## Maintenance Commands

Stop GitLab:
```bash
docker compose down
```

Backup GitLab:
```bash
docker compose exec gitlab gitlab-backup create
```

View logs:
```bash
docker compose logs -f gitlab
```

## System Requirements

Minimum recommended specifications for production use:
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB

## Troubleshooting

If GitLab fails to start:
1. Check logs: `docker compose logs -f gitlab`
2. Ensure sufficient system resources
3. Verify all ports are available
4. Check file permissions in mounted volumes

For persistent permission issues:
```bash
sudo chown -R 998:998 gitlab/
```

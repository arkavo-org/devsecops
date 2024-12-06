# DevSecOps
DevSecOps LangGraph

## Prerequisites

- `uv` https://docs.astral.sh/uv/getting-started/installation/

## Setup

### .env

```dotenv
ANTHROPIC_API_KEY=abc
GITLAB_URL=http://gitlab.localhost
GITLAB_REPOSITORY=def
OPENBAO_URL=http://gitlab.localhost:8020
OPENBAO_TOKEN=ghi
```

### Ollama

To start on another machine

```shell
OLLAMA_HOST="0.0.0.0" ollama serve
```

```shell
ollama install llama3.2
ollama install deepseek-coder-v2
```

### LangGraph

```shell
uv add langgraph langsmith langchain_anthropic
```

### Tools

```shell
uv add langchain_community duckduckgo-search langchain-ollama python-gitlab docker
```

### Patch

```shell
uv add urllib3==1.26.5
uv add langgraph==0.2.50
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


Graph

```mermaid
graph TD
    %% Current System Structure

    START((START))
    haiku[haiku]
    llama[llama]
    deepseek[deepseek]
    tools[tools]
    gitlab[gitlab]
    secrets[secrets]
    END((END))

    %% Styling
    classDef default fill:#bbf,stroke:#333,stroke-width:1px;
    classDef router fill:#f9f,stroke:#333,stroke-width:2px;
    classDef eend fill:#f96,stroke:#333,stroke-width:2px;
    classDef sstart fill:#9f9,stroke:#333,stroke-width:2px;

    class START sstart;
    class END eend;
    class haiku,llama,deepseek,tools,gitlab,secrets default;
```
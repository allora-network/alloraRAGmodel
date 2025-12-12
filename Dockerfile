FROM python:3.11-slim

WORKDIR /app

# Install Node.js (required for MCP server)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Copy and install Python dependencies
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Install MCP server from GitHub Packages
# Pass GITHUB_TOKEN as build arg: docker build --build-arg GITHUB_TOKEN=ghp_xxx .
ARG GITHUB_TOKEN
RUN --mount=type=secret,id=github_token \
    if [ -f /run/secrets/github_token ]; then \
      TOKEN=$(cat /run/secrets/github_token); \
    elif [ -n "$GITHUB_TOKEN" ]; then \
      TOKEN=$GITHUB_TOKEN; \
    else \
      echo "Warning: No GitHub token provided, skipping MCP package install"; \
      exit 0; \
    fi && \
    echo "@allora-network:registry=https://npm.pkg.github.com" > ~/.npmrc && \
    echo "//npm.pkg.github.com/:_authToken=$TOKEN" >> ~/.npmrc && \
    npm install -g @allora-network/wizard-mcp && \
    rm -f ~/.npmrc

# Copy application code
COPY *.py .

RUN chown -R 1000:1000 /app

CMD ["python", "main.py"]

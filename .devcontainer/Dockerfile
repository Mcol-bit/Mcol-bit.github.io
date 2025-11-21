# Dockerfile for a Python Development Container
# This uses the official Python 3.11 image as a base
FROM python:3.11-slim-bookworm

# Install necessary system packages (like git for cloning repos and curl for health checks)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pre-commit hooks utility globally
RUN pip install --no-cache-dir pre-commit

# Set the working directory inside the container
WORKDIR /workspaces/my-python-project

# (Optional) Copy dependency files and install them early for better caching
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# The container will run as the 'vscode' user by default when used with Codespaces/VS Code Remote
# Add any specific setup needed here. For now, we rely on the base image's setup.

# Set environment variables for non-buffered output and using virtual environments
ENV PYTHONUNBUFFERED 1
ENV VIRTUAL_ENV /usr/local/venv
ENV PATH="/usr/local/venv/bin:$PATH"

# Create a virtual environment and activate it (optional, but good practice)
# RUN python -m venv $VIRTUAL_ENV

# Note: The devcontainer.json will handle package installation (e.g., pip install -r requirements.txt)
# after the container is built and the workspace is mounted.

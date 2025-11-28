#!/bin/bash

# 1. Allow local connections to X server (works for X11 and XWayland)
xhost +local:docker > /dev/null 2>&1

# 2. Get current user ID and Group ID
# UID is readonly in bash, so we use USER_ID
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

# 3. Run via Docker Compose
docker compose up --build
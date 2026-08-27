FROM python:3.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# Keep uv independent from the Autoware image and pin the tool version so the
# project lock file is interpreted consistently across developer machines.
COPY --from=ghcr.io/astral-sh/uv:0.6.17 /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/lidar-rl-venv \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/lidar-rl-venv \
    PATH=/opt/lidar-rl-venv/bin:${PATH}

# Named volumes inherit these permissions on first creation. The container is
# intentionally run as the host UID/GID, which is not known at image build time.
RUN mkdir -p \
        /opt/lidar-rl-venv \
        /opt/uv-cache \
        /tmp/lidar-rl-home \
        /workspace \
    && chmod 1777 \
        /opt/lidar-rl-venv \
        /opt/uv-cache \
        /tmp/lidar-rl-home

WORKDIR /workspace/aichallenge/ml_workspace/lidar_racing_rl

CMD ["bash"]

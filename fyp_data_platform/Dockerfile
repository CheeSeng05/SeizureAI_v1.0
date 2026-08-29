FROM node:22-alpine AS frontend
WORKDIR /build/frontend
COPY fyp_data_platform/frontend/package.json ./
COPY fyp_data_platform/frontend/pnpm-lock.yaml ./
COPY fyp_data_platform/frontend/pnpm-workspace.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile
COPY fyp_data_platform/frontend/ ./
RUN pnpm run build

FROM python:3.12-slim AS application
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY FYPModels /app/FYPModels
COPY fyp_data_platform /app/fyp_data_platform
COPY --from=frontend /build/frontend/dist /app/fyp_data_platform/frontend/dist
RUN python -m pip install --upgrade pip && \
    python -m pip install "/app/fyp_data_platform[ui]" "/app/FYPModels[all]"
WORKDIR /app/fyp_data_platform
RUN chmod +x docker/entrypoint.sh
EXPOSE 8080
ENTRYPOINT ["docker/entrypoint.sh"]

FROM application AS gpu
RUN python -m pip install --upgrade --force-reinstall \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128
ENV FYP_COMPUTE_PROFILE=nvidia-cuda \
    FYP_REQUIRE_CUDA=1

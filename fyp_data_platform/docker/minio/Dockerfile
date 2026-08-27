# Build the final security-fixed MinIO Community release from its AGPLv3 source.
# The upstream project stopped distributing maintained community binaries in 2026.
ARG MINIO_VERSION=RELEASE.2025-10-15T17-29-55Z

FROM golang:1.24.8-alpine AS build
ARG MINIO_VERSION
ENV CGO_ENABLED=0
RUN go install "github.com/minio/minio@${MINIO_VERSION}"

FROM alpine:3.22
RUN apk add --no-cache ca-certificates wget \
    && addgroup -S minio \
    && adduser -S -G minio minio \
    && mkdir -p /data \
    && chown minio:minio /data
COPY --from=build /go/bin/minio /usr/local/bin/minio
USER minio
EXPOSE 9000 9001
ENTRYPOINT ["minio"]

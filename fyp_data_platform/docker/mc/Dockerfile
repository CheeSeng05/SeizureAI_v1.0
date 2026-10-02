# Build the pinned upstream client without depending on MinIO's image registries.
FROM golang:1.24.8-alpine AS build
ARG MC_VERSION=RELEASE.2025-08-13T08-35-41Z
ENV CGO_ENABLED=0
RUN go install "github.com/minio/mc@${MC_VERSION}"

FROM alpine:3.22
RUN apk add --no-cache ca-certificates
COPY --from=build /go/bin/mc /usr/local/bin/mc
ENTRYPOINT ["mc"]

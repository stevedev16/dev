# Multi-stage build for Go services
FROM golang:1.21-alpine AS builder

# Install git and ca-certificates
RUN apk add --no-cache git ca-certificates tzdata

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Build argument for service path
ARG SERVICE_PATH

# Copy source code
COPY ${SERVICE_PATH}/ ./

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags='-w -s -extldflags "-static"' \
    -o main .

# Production stage
FROM scratch

# Copy ca-certificates from builder
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy timezone data
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Copy the binary
COPY --from=builder /app/main /main

# Health check using wget (we need to use a base image with wget for this)
# For scratch images, health checks need to be handled externally

# Expose port
EXPOSE 8001
EXPOSE 8002

# Run the binary
ENTRYPOINT ["/main"]

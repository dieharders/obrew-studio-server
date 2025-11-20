#!/bin/bash
# Post-install script for macOS to install trusted SSL certificates
# This script runs with elevated privileges.
# This script is NOT currently used, you may delete. Here for development use.

set -e

# Get the directory where the app is installed
if [ -n "$MEIPASS" ]; then
    # Running from PyInstaller bundle
    APP_DIR="$MEIPASS"
else
    # Running from source
    APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

MKCERT_BIN="$APP_DIR/certs/mkcert-darwin-$(uname -m)"
CERT_DIR="$APP_DIR/public"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

# Validate paths before use
if [ ! -d "$APP_DIR" ]; then
    echo "[Obrew Studio] Error: Application directory not found: $APP_DIR"
    exit 1
fi

# Validate that paths don't contain suspicious characters
if [[ "$CERT_DIR" =~ [^a-zA-Z0-9/_.\ -] ]]; then
    echo "[Obrew Studio] Error: Certificate directory path contains invalid characters"
    exit 1
fi

# Create certificate directory if it doesn't exist
mkdir -p "$CERT_DIR" || {
    echo "[Obrew Studio] Error: Failed to create certificate directory"
    exit 1
}

echo "[Obrew Studio] Installing SSL certificates..."

# Check if mkcert binary exists
if [ ! -f "$MKCERT_BIN" ]; then
    # Try alternative architecture
    if [ "$(uname -m)" = "arm64" ]; then
        MKCERT_BIN="$APP_DIR/certs/mkcert-darwin-amd64"
    else
        MKCERT_BIN="$APP_DIR/certs/mkcert-darwin-arm64"
    fi

    if [ ! -f "$MKCERT_BIN" ]; then
        echo "[Obrew Studio] Error: mkcert binary not found"
        exit 1
    fi
fi

# Make mkcert executable
chmod +x "$MKCERT_BIN"

# Install CA certificate (requires sudo)
echo "[Obrew Studio] Installing certificate authority (password required)..."
"$MKCERT_BIN" -install || {
    echo "[Obrew Studio] Failed to install CA certificate"
    exit 1
}

# Generate localhost certificates
echo "[Obrew Studio] Generating localhost certificates..."
"$MKCERT_BIN" -cert-file "$CERT_FILE" -key-file "$KEY_FILE" localhost 127.0.0.1 ::1 || {
    echo "[Obrew Studio] Failed to generate certificates"
    exit 1
}

echo "[Obrew Studio] SSL certificates installed successfully!"
exit 0

#!/bin/bash
# Post-install script for macOS .pkg installer
# This runs during installation with administrator privileges
# and installs SSL certificates system-wide

set -e

echo "[Obrew Studio Installer] Starting certificate installation..."

# $2 is the installation target directory (usually /Applications)
INSTALL_DIR="$2"
APP_PATH="$INSTALL_DIR/Obrew-Studio.app/Contents/MacOS"
DEPS_DIR="$APP_PATH/_deps"

# Paths to mkcert and certificate directory
MKCERT_BIN="$DEPS_DIR/certs/mkcert-darwin-$(uname -m)"
CERT_DIR="$DEPS_DIR/public"

# Validate installation directory
if [ ! -d "$APP_PATH" ]; then
    echo "[Obrew Studio Installer] Error: Application not found at $APP_PATH"
    exit 1
fi

# Check for mkcert binary
if [ ! -f "$MKCERT_BIN" ]; then
    # Try alternative architecture
    if [ "$(uname -m)" = "arm64" ]; then
        MKCERT_BIN="$DEPS_DIR/certs/mkcert-darwin-amd64"
    else
        MKCERT_BIN="$DEPS_DIR/certs/mkcert-darwin-arm64"
    fi

    if [ ! -f "$MKCERT_BIN" ]; then
        echo "[Obrew Studio Installer] Warning: mkcert binary not found, skipping certificate installation"
        exit 0  # Don't fail the installation
    fi
fi

# Make mkcert executable
chmod +x "$MKCERT_BIN"

# Create certificate directory if needed
mkdir -p "$CERT_DIR"

echo "[Obrew Studio Installer] Installing certificate authority..."

# Install CA certificate (we already have elevated privileges from the installer)
# Note: This will install to the system keychain
"$MKCERT_BIN" -install || {
    echo "[Obrew Studio Installer] Warning: Failed to install CA certificate"
    exit 0  # Don't fail the installation
}

echo "[Obrew Studio Installer] Generating localhost certificates..."

# Generate localhost certificates
"$MKCERT_BIN" -cert-file "$CERT_DIR/cert.pem" -key-file "$CERT_DIR/key.pem" localhost 127.0.0.1 ::1 || {
    echo "[Obrew Studio Installer] Warning: Failed to generate certificates"
    exit 0  # Don't fail the installation
}

# Set proper permissions on generated certificates
chmod 644 "$CERT_DIR/cert.pem"
chmod 644 "$CERT_DIR/key.pem"

echo "[Obrew Studio Installer] SSL certificates installed successfully!"
exit 0

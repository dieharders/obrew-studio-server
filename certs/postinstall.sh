#!/bin/bash
# Post-install script for macOS .pkg installer
# This runs during installation with administrator privileges
# and installs SSL certificates system-wide

set -e

echo "[Obrew Studio Installer] Starting certificate installation..."

# $2 is the installation target directory (usually /Applications)
INSTALL_DIR="$2"
APP_BUNDLE="$INSTALL_DIR/Obrew-Studio.app"
# PyInstaller places dependencies in Contents/Frameworks on macOS
FRAMEWORKS_DIR="$APP_BUNDLE/Contents/Frameworks"

# Get the actual logged-in user (not root, since installer runs with elevated privileges)
CONSOLE_USER=$(stat -f '%Su' /dev/console)
USER_HOME=$(eval echo ~"$CONSOLE_USER")

# Paths to mkcert binary (bundled in Frameworks/certs/)
MKCERT_BIN="$FRAMEWORKS_DIR/certs/mkcert-darwin-$(uname -m)"
# Create certificates in Application Support (where app_path() looks on macOS)
# This location is writable and persists across app updates
CERT_DIR="$USER_HOME/Library/Application Support/Obrew-Studio/public"

echo "[Obrew Studio Installer] Installing for user: $CONSOLE_USER"
echo "[Obrew Studio Installer] Certificate directory: $CERT_DIR"

# Validate installation directory
if [ ! -d "$APP_BUNDLE" ]; then
    echo "[Obrew Studio Installer] Error: Application not found at $APP_BUNDLE"
    exit 1
fi

# Check for mkcert binary
if [ ! -f "$MKCERT_BIN" ]; then
    # Try alternative architecture
    if [ "$(uname -m)" = "arm64" ]; then
        MKCERT_BIN="$FRAMEWORKS_DIR/certs/mkcert-darwin-amd64"
    else
        MKCERT_BIN="$FRAMEWORKS_DIR/certs/mkcert-darwin-arm64"
    fi

    if [ ! -f "$MKCERT_BIN" ]; then
        echo "[Obrew Studio Installer] Warning: mkcert binary not found at $MKCERT_BIN"
        echo "[Obrew Studio Installer] Skipping certificate installation - will use bundled fallback certs"
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

# Set proper permissions and ownership on generated certificates
chmod 644 "$CERT_DIR/cert.pem"
chmod 644 "$CERT_DIR/key.pem"
# Change ownership to the actual user (since we're running as root)
chown -R "$CONSOLE_USER" "$USER_HOME/Library/Application Support/Obrew-Studio"

echo "[Obrew Studio Installer] SSL certificates installed successfully!"
exit 0

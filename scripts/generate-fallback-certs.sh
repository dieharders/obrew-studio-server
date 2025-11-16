#!/bin/bash
# Generate self-signed fallback SSL certificates for localhost
# These are used when mkcert installation fails or is declined

set -e

CERT_DIR="backends/ui/public"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

# Create directory if it doesn't exist
mkdir -p "$CERT_DIR"

echo "Generating self-signed fallback SSL certificates..."

# Generate self-signed certificate
# - RSA 4096-bit key
# - Valid for 100 years (36500 days)
# - Subject: CN=localhost (for localhost HTTPS)
openssl req -x509 -newkey rsa:4096 -nodes \
  -out "$CERT_FILE" \
  -keyout "$KEY_FILE" \
  -days 36500 \
  -subj "/C=US/ST=State/L=City/O=ObrewStudio/CN=localhost" \
  2>/dev/null

if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
    echo "✓ Self-signed fallback certificates generated successfully"
    echo "  Certificate: $CERT_FILE"
    echo "  Private Key: $KEY_FILE"
    echo ""
    echo "Note: These are fallback certificates. Browsers will show security warnings."
    echo "Trusted mkcert certificates (no warnings) are installed during app installation."
else
    echo "❌ Error: Failed to generate certificates"
    exit 1
fi

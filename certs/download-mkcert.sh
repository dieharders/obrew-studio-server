#!/bin/bash
# Download mkcert binaries for all platforms
# Version: 1.4.4

set -e

MKCERT_VERSION="v1.4.4"
BUNDLED_DIR="certs"
BASE_URL="https://github.com/FiloSottile/mkcert/releases/download/${MKCERT_VERSION}"

# Create certs directory if it doesn't exist
mkdir -p "$BUNDLED_DIR"

# SHA256 checksums for mkcert v1.4.4
get_checksum() {
    case "$1" in
        "mkcert-darwin-arm64")
            echo "c8af0df44bce04359794dad8ea28d750437411d632748049d08644ffb66a60c6"
            ;;
        "mkcert-darwin-amd64")
            echo "a32dfab51f1845d51e810db8e47dcf0e6b51ae3422426514bf5a2b8302e97d4e"
            ;;
        "mkcert-windows-amd64.exe")
            echo "d2660b50a9ed59eada480750561c96abc2ed4c9a38c6a24d93e30e0977631398"
            ;;
        "mkcert-linux-amd64")
            echo "6d31c65b03972c6dc4a14ab429f2928300518b26503f58723e532d1b0a3bbb52"
            ;;
    esac
}

# Download and verify each binary
for binary in "mkcert-darwin-arm64" "mkcert-darwin-amd64" "mkcert-windows-amd64.exe" "mkcert-linux-amd64"; do
    filepath="$BUNDLED_DIR/$binary"

    # Skip if already exists and verified
    if [ -f "$filepath" ]; then
        echo "✓ $binary already exists, verifying checksum..."
        if command -v sha256sum &> /dev/null; then
            actual_checksum=$(sha256sum "$filepath" | awk '{print $1}')
        elif command -v shasum &> /dev/null; then
            actual_checksum=$(shasum -a 256 "$filepath" | awk '{print $1}')
        else
            echo "⚠️  No checksum utility found, skipping verification"
            continue
        fi

        expected_checksum=$(get_checksum "$binary")
        if [ "$actual_checksum" = "$expected_checksum" ]; then
            echo "✓ $binary checksum verified"
            continue
        else
            echo "❌ $binary checksum mismatch, re-downloading..."
            rm "$filepath"
        fi
    fi

    echo "Downloading $binary..."
    url="${BASE_URL}/mkcert-${MKCERT_VERSION}-${binary/mkcert-/}"

    if command -v curl &> /dev/null; then
        curl -L -o "$filepath" "$url"
    elif command -v wget &> /dev/null; then
        wget -O "$filepath" "$url"
    else
        echo "❌ Error: Neither curl nor wget found. Please install one."
        exit 1
    fi

    # Verify checksum
    echo "Verifying $binary..."
    if command -v sha256sum &> /dev/null; then
        actual_checksum=$(sha256sum "$filepath" | awk '{print $1}')
    elif command -v shasum &> /dev/null; then
        actual_checksum=$(shasum -a 256 "$filepath" | awk '{print $1}')
    else
        echo "⚠️  Warning: No checksum utility found, cannot verify download"
        continue
    fi

    expected_checksum=$(get_checksum "$binary")
    if [ "$actual_checksum" = "$expected_checksum" ]; then
        echo "✓ $binary verified successfully"
    else
        echo "❌ Error: Checksum mismatch for $binary"
        echo "Expected: $expected_checksum"
        echo "Got: $actual_checksum"
        rm "$filepath"
        exit 1
    fi

    # Make executable (Unix systems)
    if [[ "$binary" != *.exe ]]; then
        chmod +x "$filepath"
        echo "✓ Made $binary executable"
    fi
done

echo ""
echo "✅ All mkcert binaries downloaded and verified successfully!"
echo ""
ls -lh "$BUNDLED_DIR"/mkcert-*

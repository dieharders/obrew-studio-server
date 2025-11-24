#!/bin/bash

# Custom packaging script for macOS
# This bypasses CPack's productbuild issues and creates a proper .pkg installer

set -e

# Configuration
APP_NAME="Obrew-Studio"
APP_BUNDLE="$APP_NAME.app"
PKG_IDENTIFIER="com.OpenBrewAi.$APP_NAME"
PKG_VERSION="0.9.0"
PKG_NAME="$APP_NAME.macOS.Setup.pkg"

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$PROJECT_ROOT/dist"
OUTPUT_DIR="$PROJECT_ROOT/output"
TEMP_DIR="$PROJECT_ROOT/build_temp/custom_pkg"

# Clean and create directories
echo "Creating packaging directories..."
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR/payload/Applications"
mkdir -p "$OUTPUT_DIR"

# Check if app bundle exists
if [ ! -d "$DIST_DIR/$APP_BUNDLE" ]; then
    echo "Error: App bundle not found at $DIST_DIR/$APP_BUNDLE"
    exit 1
fi

# Copy app bundle to payload
echo "Copying app bundle to payload..."
cp -R "$DIST_DIR/$APP_BUNDLE" "$TEMP_DIR/payload/Applications/"

# Check size
echo "App bundle size:"
du -sh "$TEMP_DIR/payload/Applications/$APP_BUNDLE"

# Create the component package
echo "Creating component package..."
pkgbuild \
    --root "$TEMP_DIR/payload" \
    --identifier "$PKG_IDENTIFIER" \
    --version "$PKG_VERSION" \
    --install-location "/" \
    "$TEMP_DIR/component.pkg"

# Check component package size
echo "Component package size:"
ls -lh "$TEMP_DIR/component.pkg"

# Create distribution XML
echo "Creating distribution XML..."
cat > "$TEMP_DIR/distribution.xml" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="1">
    <title>$APP_NAME</title>
    <organization>OpenBrewAi</organization>
    <domains enable_localSystem="true"/>
    <options customize="never" require-scripts="true" rootVolumeOnly="true"/>
    <pkg-ref id="$PKG_IDENTIFIER">
        <bundle-version/>
    </pkg-ref>
    <choices-outline>
        <line choice="default">
            <line choice="$PKG_IDENTIFIER"/>
        </line>
    </choices-outline>
    <choice id="default"/>
    <choice id="$PKG_IDENTIFIER" visible="false">
        <pkg-ref id="$PKG_IDENTIFIER"/>
    </choice>
    <pkg-ref id="$PKG_IDENTIFIER" version="$PKG_VERSION" onConclusion="none">component.pkg</pkg-ref>
</installer-gui-script>
EOF

# Create the final product package
echo "Creating final product package..."
productbuild \
    --distribution "$TEMP_DIR/distribution.xml" \
    --package-path "$TEMP_DIR" \
    --version "$PKG_VERSION" \
    "$OUTPUT_DIR/$PKG_NAME"

# Check final package
echo "Final package created:"
ls -lh "$OUTPUT_DIR/$PKG_NAME"

# Verify package contents (optional)
if command -v pkgutil &> /dev/null; then
    echo ""
    echo "Package contents verification:"
    pkgutil --payload-files "$OUTPUT_DIR/$PKG_NAME" | head -20
fi

echo ""
echo "Packaging complete: $OUTPUT_DIR/$PKG_NAME"
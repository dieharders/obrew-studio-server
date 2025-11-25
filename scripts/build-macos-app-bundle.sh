#!/bin/bash
set -e  # Exit on error

WORKSPACE_DIR="${1:-$GITHUB_WORKSPACE}"

echo "=== macOS Post-Build Processing ==="
echo "Working directory: $WORKSPACE_DIR"

# On macOS, PyInstaller with --onedir --windowed creates an app bundle
# The app bundle should be in dist/Obrew-Studio.app
echo "macOS build completed"

# Ensure the app bundle exists
if [ ! -d "$WORKSPACE_DIR/dist/Obrew-Studio.app" ]; then
  echo "WARNING: App bundle not found at dist/Obrew-Studio.app"
  echo "Checking what PyInstaller created:"
  ls -la "$WORKSPACE_DIR/dist/"
  echo "Checking build directory:"
  ls -la "$WORKSPACE_DIR/build/" 2>/dev/null || echo "No build directory"

  # If PyInstaller created a directory (--onedir without proper app bundle)
  if [ -d "$WORKSPACE_DIR/dist/Obrew-Studio" ]; then
    echo "Found Obrew-Studio directory, converting to app bundle structure..."
    mkdir -p "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/MacOS"
    mkdir -p "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/Resources"
    # Move all contents including _deps folder
    mv "$WORKSPACE_DIR/dist/Obrew-Studio/"* "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/MacOS/"
    rmdir "$WORKSPACE_DIR/dist/Obrew-Studio"

    # Create minimal Info.plist
    cat > "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>Obrew-Studio</string>
    <key>CFBundleIdentifier</key>
    <string>com.openbrewai.obrew-studio</string>
    <key>CFBundleName</key>
    <string>Obrew Studio</string>
    <key>CFBundleVersion</key>
    <string>0.9.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
EOF
  # If PyInstaller created just an executable (fallback for --onefile mode)
  elif [ -f "$WORKSPACE_DIR/dist/Obrew-Studio" ]; then
    echo "Found executable, creating app bundle structure..."
    mkdir -p "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/MacOS"
    mkdir -p "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/Resources"
    mv "$WORKSPACE_DIR/dist/Obrew-Studio" "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/MacOS/"

    # Create minimal Info.plist
    cat > "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>Obrew-Studio</string>
    <key>CFBundleIdentifier</key>
    <string>com.openbrewai.obrew-studio</string>
    <key>CFBundleName</key>
    <string>Obrew Studio</string>
    <key>CFBundleVersion</key>
    <string>0.9.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
EOF
  fi
fi

echo "macOS app bundle prepared"

# Debug output
echo ""
echo "=== POST-BUILD DEBUGGING ==="
echo "Workspace directory:"
ls -al "$WORKSPACE_DIR"
echo ""
echo "Build directory:"
ls -al "$WORKSPACE_DIR/build/" 2>/dev/null || echo "No build directory"
echo ""
echo "Dist directory contents:"
ls -la "$WORKSPACE_DIR/dist/"
echo ""

echo "=== macOS App Bundle Verification ==="
if [ -d "$WORKSPACE_DIR/dist/Obrew-Studio.app" ]; then
  echo "✓ App bundle exists at dist/Obrew-Studio.app"
  echo "App bundle structure:"
  find "$WORKSPACE_DIR/dist/Obrew-Studio.app" -maxdepth 4 -type f -o -type d | head -50
  echo ""
  echo "App bundle size:"
  du -sh "$WORKSPACE_DIR/dist/Obrew-Studio.app"
  echo ""
  echo "Info.plist contents:"
  cat "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/Info.plist" 2>/dev/null || echo "No Info.plist found"
  echo ""
  echo "Contents/MacOS directory:"
  ls -la "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/MacOS/" 2>/dev/null || echo "No MacOS directory"
else
  echo "✗ App bundle NOT found at dist/Obrew-Studio.app"
  echo "Checking what files exist in dist/:"
  find "$WORKSPACE_DIR/dist/" -type f | head -20
  exit 1
fi

echo ""
echo "=== App bundle creation complete ==="

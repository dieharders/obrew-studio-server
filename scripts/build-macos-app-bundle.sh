#!/bin/bash
set -e  # Exit on error

WORKSPACE_DIR="${1:-$GITHUB_WORKSPACE}"

echo "=== macOS Post-Build Processing ==="
echo "Working directory: $WORKSPACE_DIR"

# On macOS, PyInstaller with --onefile --windowed creates an app bundle
# The app bundle should be in dist/Obrew-Studio.app
echo "macOS build completed"

# Check what PyInstaller created
echo "Checking what PyInstaller created:"
ls -la "$WORKSPACE_DIR/dist/"

# Verify PyInstaller created the app bundle
if [ ! -d "$WORKSPACE_DIR/dist/Obrew-Studio.app" ]; then
  echo "ERROR: PyInstaller did not create Obrew-Studio.app bundle"
  exit 1
fi

# Copy llama.cpp binaries into the app bundle
# (PyInstaller handles the executable and dependencies, but we need to add llama.cpp)
echo "Copying llama.cpp binaries to app bundle..."
mkdir -p "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/Frameworks/servers/llama.cpp"
cp -r "$WORKSPACE_DIR/servers/llama.cpp/"* "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/Frameworks/servers/llama.cpp/"
echo "Copied llama.cpp files:"
ls -lh "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/Frameworks/servers/llama.cpp/"

# Copy executable if not already in the app bundle
# if [ -f "$WORKSPACE_DIR/dist/Obrew-Studio/Obrew-Studio" ]; then
#   echo "Moving executable to app bundle..."
#   mv "$WORKSPACE_DIR/dist/Obrew-Studio/Obrew-Studio" "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/MacOS/"
# fi

# Copy dependencies if they exist
# if [ -d "$WORKSPACE_DIR/dist/Obrew-Studio/_deps" ]; then
#   echo "Moving dependencies to app bundle..."
#   mv "$WORKSPACE_DIR/dist/Obrew-Studio/_deps/"* "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/Frameworks/"
# fi

# Copy and rename .env.example to .env then place in Frameworks/
cp "$WORKSPACE_DIR/.env.example" "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/Frameworks/.env"

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
  find "$WORKSPACE_DIR/dist/Obrew-Studio.app" -maxdepth 3 -type f -o -type d | head -30
  echo ""
  echo "App bundle size:"
  du -sh "$WORKSPACE_DIR/dist/Obrew-Studio.app"
  echo ""
  echo "Info.plist contents:"
  cat "$WORKSPACE_DIR/dist/Obrew-Studio.app/Contents/Info.plist" 2>/dev/null || echo "No Info.plist found"
else
  echo "✗ App bundle NOT found at dist/Obrew-Studio.app"
  echo "Checking what files exist in dist/:"
  find "$WORKSPACE_DIR/dist/" -type f | head -20
  exit 1
fi

echo ""
echo "=== App bundle creation complete ==="

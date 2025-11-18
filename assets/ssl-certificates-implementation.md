# SSL Certificate Implementation Guide

## Overview

Obrew Studio now supports trusted HTTPS connections using a hybrid certificate approach:

- **Preferred**: mkcert-generated trusted certificates (no browser warnings)
- **Fallback**: Bundled self-signed certificates (browser warnings acceptable)

## How It Works

### Windows (Installer-Based)

1. **During Installation** (Inno Setup with admin privileges):

   - User grants UAC permission for admin install
   - Installer runs `mkcert -install` (installs CA to Windows Certificate Store)
   - Installer runs `mkcert` to generate localhost certificates
   - Certificates saved to `%APPDATA%\Obrew-Studio\_deps\backends\ui\public\`
   - **Result**: Zero browser warnings ✅

2. **If User Declines Admin**:
   - Installation continues with bundled self-signed certificates
   - User sees one-time browser security warning
   - User clicks "Advanced" → "Proceed to localhost"
   - **Result**: App still works, minor UX friction ⚠️

### macOS (DMG-Based)

1. **On First Launch**:

   - App detects no mkcert certificates exist
   - Prompts for password via macOS native dialog
   - Runs `certs/install_certificates_macos.sh`
   - Script executes `mkcert -install` and generates certificates
   - Certificates saved to app bundle `Contents/Resources/backends/ui/public/`
   - **Result**: Zero browser warnings on subsequent launches ✅

2. **If User Declines Password**:
   - Falls back to bundled self-signed certificates
   - Same UX as Windows fallback

## Implementation Details

### Mkcert binaries downloaded during build process and included.

#### mkcert Binaries (`bundled/`)

- `mkcert-darwin-arm64` (macOS Apple Silicon) - 5.1 MB
- `mkcert-darwin-amd64` (macOS Intel) - 4.9 MB
- `mkcert-windows-amd64.exe` (Windows) - 4.7 MB
- `mkcert-linux-amd64` (Linux - future) - 4.6 MB

**For Developers:**

```bash
pnpm run download-mkcert
```

## User Experience

### Windows Installation Flow

1. User downloads `Obrew-Studio.WIN.Setup.exe`
2. Double-clicks installer
3. UAC prompt: "Do you want to allow this app to make changes?"
4. User clicks "Yes"
5. Installer shows progress:
   - "Installing SSL certificate authority..." ✓
   - "Generating localhost certificates..." ✓
   - "Installing Obrew Studio..." ✓
6. App launches
7. Browser opens `https://localhost:8008` - **No warnings!** ✅

### macOS Installation Flow

1. User downloads `Obrew-Studio.dmg`
2. Drags app to Applications folder
3. Launches app
4. macOS dialog: "Obrew Studio wants to make changes. Enter password:"
5. User enters password
6. Terminal briefly shows: "Installing SSL certificates..." ✓
7. App launches
8. Browser opens `https://localhost:8008` - **No warnings!** ✅

### Fallback Flow (User Declines)

1. Installation completes with bundled certificates
2. App launches normally
3. Browser shows: "Your connection is not private"
4. User clicks "Advanced"
5. User clicks "Proceed to localhost (unsafe)"
6. App works normally
7. Browser remembers choice (no warnings on subsequent visits)

### Potential Improvements

1. **Auto-regeneration**: Detect expired certificates and regenerate
2. **Settings UI**: Add "Reinstall Certificates" button in app settings
3. **Linux Support**: Implement same flow for Linux `.deb` / `.rpm` installers
4. **Silent Installation**: Research if admin elevation can be silent (Windows)
5. **Uninstall Hook**: Remove CA certificate when app is uninstalled

## References

- [mkcert GitHub](https://github.com/FiloSottile/mkcert)
- [Inno Setup Documentation](https://jrsoftware.org/isinfo.php)
- [CMake CPack Documentation](https://cmake.org/cmake/help/latest/module/CPack.html)
- [MDN: Mixed Content](https://developer.mozilla.org/en-US/docs/Web/Security/Mixed_content)

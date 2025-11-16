# Bundled Binaries & Certificates

This directory contains platform-specific binaries and SSL certificates used by Obrew Studio.

## mkcert

mkcert is a simple tool for making locally-trusted development certificates. It requires no configuration and automatically installs a local CA certificate to the system trust store.

### Download & Generation Instructions

**Nothing is committed to git** - everything is generated during build:

**For Developers:**

```bash
pnpm run download-mkcert
```

This single command will:

1. Download all mkcert binaries for all platforms
2. Generate self-signed fallback SSL certificates

**For CI/CD:**
The GitHub Actions workflow automatically runs the download script before building.

### What Gets Generated

**mkcert Binaries:**

- `mkcert-darwin-arm64` - macOS ARM64 (Apple Silicon M1/M2/M3)
- `mkcert-darwin-amd64` - macOS Intel (x86_64)
- `mkcert-linux-amd64` - Linux AMD64
- `mkcert-windows-amd64.exe` - Windows AMD64

**Self-Signed Fallback Certificates:**

- `backends/ui/public/cert.pem` - Public certificate
- `backends/ui/public/key.pem` - Private key

These fallback certificates are used when mkcert installation fails or is declined by the user.

### Versions

- **mkcert**: v1.4.4
- **OpenSSL**: System version (for cert generation)

### Sources

- mkcert: https://github.com/FiloSottile/mkcert
- OpenSSL: https://www.openssl.org

### Licenses

- mkcert: BSD 3-Clause License
- OpenSSL: Apache License 2.0

### Usage

**For End Users:**
No action needed. The installer handles everything:

1. **Windows**: Installer runs mkcert with admin privileges
2. **macOS**: First launch prompts for password to install mkcert. This happenns during runtime since .dmg cannot do this during app install.
3. **Fallback**: If mkcert fails, self-signed certificates are used (browser shows warning)

**For Developers:**
Run `pnpm run download-mkcert` once to set up your local development environment with all necessary binaries and certificates.

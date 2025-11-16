# Bundled Binaries

This directory contains platform-specific binaries bundled with Obrew Studio.

## mkcert

mkcert is a simple tool for making locally-trusted development certificates. It requires no configuration and automatically installs a local CA certificate to the system trust store.

### Included Binaries

- `mkcert-darwin-arm64` - macOS ARM64 (Apple Silicon M1/M2/M3)
- `mkcert-darwin-amd64` - macOS Intel (x86_64)
- `mkcert-linux-amd64` - Linux AMD64
- `mkcert-windows-amd64.exe` - Windows AMD64

### Version

mkcert v1.4.4

### Source

https://github.com/FiloSottile/mkcert

### License

BSD 3-Clause License - see https://github.com/FiloSottile/mkcert/blob/master/LICENSE

### Usage

These binaries are automatically used by Obrew Studio's certificate manager to install trusted SSL certificates for localhost HTTPS connections. No manual intervention is required.

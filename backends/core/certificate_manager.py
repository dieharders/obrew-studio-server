import subprocess
import sys
import platform
from pathlib import Path


class CertificateManager:
    """
    Manages SSL certificate validation for Obrew Studio.

    Note: Certificate installation is handled by the installer (Windows)
    or on first launch (macOS). This class only validates certificates exist.
    """

    def __init__(self, cert_dir: str = None):
        """
        Initialize the certificate manager.

        Args:
            cert_dir: Directory where certificates are stored (defaults to backends/ui/public)
        """
        if cert_dir:
            self.cert_dir = Path(cert_dir)
        else:
            # Default to backends/ui/public
            base_path = Path(__file__).parent.parent
            self.cert_dir = base_path / "ui" / "public"

        self.cert_file = self.cert_dir / "cert.pem"
        self.key_file = self.cert_dir / "key.pem"

    def are_certificates_valid(self) -> bool:
        """
        Check if certificate files exist and are readable.

        Returns:
            bool: True if both cert and key files exist
        """
        return self.cert_file.exists() and self.key_file.exists()

    # Check SSL certificate status and install if needed (macOS only)
    def check_and_install_ssl_certificates_macos(self):
        """
        Check if SSL certificates exist and install them if needed.

        For macOS: If no certificates exist, runs install_certificates_macos.sh
        For Windows: Certificates are installed during Inno Setup installation
        For other platforms: Uses fallback self-signed certificates
        """

        # Check if certificates already exist
        if self.are_certificates_valid():
            # Certificates exist, we're good to go
            print(
                f"[CERT] Certificate exist.",
                flush=True,
            )
            return

        # No valid certificates found - attempt installation on macOS
        if platform.system() == "Darwin":
            try:
                # Find the installation script
                if getattr(sys, "frozen", False):
                    # Running from PyInstaller bundle
                    base_dir = Path(sys._MEIPASS)
                else:
                    # Running from source
                    # This file is at: backends/core/certificate_manager.py
                    # We need to go up 2 levels to reach project root
                    base_dir = Path(__file__).parent.parent.parent

                script_path = base_dir / "certs" / "install_certificates_macos.sh"

                if not script_path.exists():
                    print(
                        f"[CERT] Warning: Certificate installation script not found at {script_path}",
                        flush=True,
                    )
                    print(
                        f"[CERT] Falling back to bundled self-signed certificates",
                        flush=True,
                    )
                    return

                print(
                    f"[CERT] No SSL certificates found. Installing trusted certificates...",
                    flush=True,
                )

                # Run the installation script
                result = subprocess.run(
                    ["/bin/bash", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=120,  # 2 minute timeout
                )

                if result.returncode == 0:
                    print(f"[CERT] SSL certificates installed successfully", flush=True)
                    return
                else:
                    print(f"[CERT] Failed to install SSL certificates:", flush=True)
                    if result.stdout:
                        print(f"[CERT] stdout: {result.stdout}", flush=True)
                    if result.stderr:
                        print(f"[CERT] stderr: {result.stderr}", flush=True)
                    print(f"[CERT] Return code: {result.returncode}", flush=True)
                    print(
                        f"[CERT] Falling back to bundled self-signed certificates",
                        flush=True,
                    )
                    return

            except subprocess.TimeoutExpired:
                print(f"[CERT] Certificate installation timed out", flush=True)
                print(
                    f"[CERT] Falling back to bundled self-signed certificates",
                    flush=True,
                )
                return
            except Exception as e:
                print(f"[CERT] Error during certificate installation: {e}", flush=True)
                print(
                    f"[CERT] Falling back to bundled self-signed certificates",
                    flush=True,
                )
                return

        # For Windows and other platforms, certificates should already be installed
        # or we'll use the bundled fallback certificates
        return

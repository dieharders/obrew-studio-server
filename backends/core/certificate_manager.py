import os
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
        self.mkcert_installed_flag = self.cert_dir / ".mkcert_installed"

    def are_certificates_valid(self) -> bool:
        """
        Check if certificate files exist and are readable.

        Returns:
            bool: True if both cert and key files exist
        """
        return self.cert_file.exists() and self.key_file.exists()

    def are_mkcert_certificates_installed(self) -> bool:
        """
        Check if mkcert-generated certificates are installed.

        Returns:
            bool: True if mkcert certificates were successfully installed by installer
        """
        return self.mkcert_installed_flag.exists() and self.are_certificates_valid()

    def get_certificate_type(self) -> str:
        """
        Determine what type of certificates are being used.

        Returns:
            str: "mkcert" if trusted certs, "self-signed" if bundled fallback, "none" if missing
        """
        if not self.are_certificates_valid():
            return "none"

        if self.are_mkcert_certificates_installed():
            return "mkcert"

        return "self-signed"

    def get_certificate_paths(self) -> dict:
        """
        Get the paths to certificate files.

        Returns:
            dict: Paths to cert and key files
        """
        return {
            "cert_file": str(self.cert_file),
            "key_file": str(self.key_file),
        }

    def check_certificate_type(self, cert_type) -> bool:
        """
        Check SSL certificate status and log which type is being used.

        Note: Certificate installation is handled by:
        - Windows: Installer (Inno Setup) with admin privileges
        - macOS: Installer creates certificates on first launch
        - Fallback: Bundled self-signed certificates (browser warnings)
        """
        if cert_type == "none":
            # print(f"{common.PRNT_APP} ❌ Error: No SSL certificates found!", flush=True)
            # print(f"{common.PRNT_APP} The app may not function correctly.", flush=True)
            return False

        elif cert_type == "mkcert":
            # print(f"{common.PRNT_APP} ✓ Using trusted mkcert certificates (no browser warnings)", flush=True)
            return True

        elif cert_type == "self-signed":
            # print(f"{common.PRNT_APP} ⚠️  Using bundled self-signed certificates", flush=True)
            # print(f"{common.PRNT_APP} Note: Browsers will show a one-time security warning", flush=True)
            # print(f"{common.PRNT_APP} You can accept the warning to proceed", flush=True)
            return True

        return True

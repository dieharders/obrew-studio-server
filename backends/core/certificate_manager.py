import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Tuple


class CertificateManager:
    """
    Manages SSL certificates using mkcert for trusted localhost HTTPS.

    This class handles:
    - Detection of platform-specific mkcert binaries
    - Installation of mkcert CA certificate to system trust store
    - Generation of localhost certificates
    - Verification of certificate installation status
    """

    def __init__(self, cert_dir: str = None):
        """
        Initialize the certificate manager.

        Args:
            cert_dir: Directory to store certificates (defaults to backends/ui/public)
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

        # Platform-specific mkcert binary paths
        self.bundled_dir = Path(__file__).parent.parent.parent / "bundled"

    def get_mkcert_binary(self) -> str:
        """
        Get the platform-specific mkcert binary path.

        Returns:
            str: Path to the mkcert binary for the current platform

        Raises:
            RuntimeError: If platform is not supported
        """
        system = platform.system()
        machine = platform.machine().lower()

        if system == "Darwin":  # macOS
            if "arm" in machine or "aarch64" in machine:
                binary = "mkcert-darwin-arm64"
            else:
                binary = "mkcert-darwin-amd64"
        elif system == "Windows":
            binary = "mkcert-windows-amd64.exe"
        elif system == "Linux":
            binary = "mkcert-linux-amd64"
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

        mkcert_path = self.bundled_dir / binary

        # Make executable on Unix-like systems
        if system in ["Darwin", "Linux"] and mkcert_path.exists():
            mkcert_path.chmod(0o755)

        return str(mkcert_path)

    def is_mkcert_available(self) -> bool:
        """
        Check if mkcert binary is available.

        Returns:
            bool: True if mkcert binary exists
        """
        try:
            mkcert_path = Path(self.get_mkcert_binary())
            return mkcert_path.exists()
        except RuntimeError:
            return False

    def is_ca_installed(self) -> bool:
        """
        Check if mkcert CA is already installed on the system.

        Returns:
            bool: True if CA is installed
        """
        return self.mkcert_installed_flag.exists()

    def are_certificates_valid(self) -> bool:
        """
        Check if certificate files exist and are readable.

        Returns:
            bool: True if both cert and key files exist
        """
        return self.cert_file.exists() and self.key_file.exists()

    def install_ca_certificate(self) -> Tuple[bool, str]:
        """
        Install mkcert CA certificate to system trust store.

        This will prompt the user for administrator/root password
        to add the CA certificate to the system trust store.

        Returns:
            Tuple[bool, str]: (Success status, Error message if failed)
        """
        if not self.is_mkcert_available():
            return False, "mkcert binary not found. Please ensure bundled binaries are present."

        mkcert = self.get_mkcert_binary()

        try:
            # Install CA (requires user permission on macOS/Windows)
            result = subprocess.run(
                [mkcert, "-install"],
                check=True,
                capture_output=True,
                text=True
            )

            print(f"mkcert CA installed: {result.stdout}")

            # Mark as installed
            self.mkcert_installed_flag.touch()
            return True, ""

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to install CA certificate: {e.stderr}"
            print(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error installing CA: {str(e)}"
            print(error_msg)
            return False, error_msg

    def generate_certificates(self) -> Tuple[bool, str]:
        """
        Generate localhost certificates using mkcert.

        Returns:
            Tuple[bool, str]: (Success status, Error message if failed)
        """
        if not self.is_mkcert_available():
            return False, "mkcert binary not found"

        mkcert = self.get_mkcert_binary()

        # Ensure certificate directory exists
        self.cert_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Generate certificates for localhost, 127.0.0.1, and ::1 (IPv6)
            result = subprocess.run(
                [
                    mkcert,
                    "-cert-file", str(self.cert_file),
                    "-key-file", str(self.key_file),
                    "localhost",
                    "127.0.0.1",
                    "::1"
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(self.cert_dir)
            )

            print(f"Certificates generated: {result.stdout}")
            return True, ""

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to generate certificates: {e.stderr}"
            print(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error generating certificates: {str(e)}"
            print(error_msg)
            return False, error_msg

    def uninstall_ca_certificate(self) -> Tuple[bool, str]:
        """
        Uninstall mkcert CA certificate from system trust store.

        Returns:
            Tuple[bool, str]: (Success status, Error message if failed)
        """
        if not self.is_mkcert_available():
            return False, "mkcert binary not found"

        mkcert = self.get_mkcert_binary()

        try:
            result = subprocess.run(
                [mkcert, "-uninstall"],
                check=True,
                capture_output=True,
                text=True
            )

            print(f"mkcert CA uninstalled: {result.stdout}")

            # Remove flag
            if self.mkcert_installed_flag.exists():
                self.mkcert_installed_flag.unlink()

            return True, ""

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to uninstall CA certificate: {e.stderr}"
            print(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error uninstalling CA: {str(e)}"
            print(error_msg)
            return False, error_msg

    def setup_certificates(self, force_reinstall: bool = False) -> Tuple[bool, str]:
        """
        Main entry point: ensures certificates are trusted and ready.

        This method will:
        1. Check if CA is already installed
        2. Install CA if needed (prompts user for password)
        3. Generate certificates

        Args:
            force_reinstall: If True, reinstall even if already installed

        Returns:
            Tuple[bool, str]: (Success status, Error message if failed)
        """
        # Check if already set up
        if not force_reinstall and self.is_ca_installed() and self.are_certificates_valid():
            print("SSL certificates already installed and valid")
            return True, ""

        print("Setting up trusted SSL certificates for localhost...")

        # Step 1: Install CA
        if not self.is_ca_installed() or force_reinstall:
            success, error = self.install_ca_certificate()
            if not success:
                return False, f"CA installation failed: {error}"

        # Step 2: Generate certificates
        success, error = self.generate_certificates()
        if not success:
            return False, f"Certificate generation failed: {error}"

        print("✅ SSL certificates installed successfully!")
        return True, ""

    def get_setup_status(self) -> dict:
        """
        Get detailed status of certificate setup.

        Returns:
            dict: Status information including:
                - mkcert_available: bool
                - ca_installed: bool
                - certificates_valid: bool
                - ready: bool (all checks passed)
        """
        mkcert_available = self.is_mkcert_available()
        ca_installed = self.is_ca_installed()
        certs_valid = self.are_certificates_valid()

        return {
            "mkcert_available": mkcert_available,
            "ca_installed": ca_installed,
            "certificates_valid": certs_valid,
            "ready": mkcert_available and ca_installed and certs_valid,
            "cert_file": str(self.cert_file),
            "key_file": str(self.key_file),
        }

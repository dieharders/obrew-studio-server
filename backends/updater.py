from typing import Dict, List
import requests
import zipfile
import os
from io import BytesIO
import platform
import GPUtil
from core.common import dep_path, get_package_json

# Conditional import for Windows-only dependencies
if platform.system() == "Windows":
    import wmi

# @TODO Perhaps make a small installer app with Tauri that has a GUI that can launch headless/non etc of app and install/notify of updates. We wouldn't need multiple app shortcuts and the download link can always point to one file for all platforms.


def get_gpu_details() -> List[Dict]:
    results = []

    # macOS - Return Metal GPU info for Apple Silicon
    if platform.system() == "Darwin":
        import subprocess

        try:
            # Get Mac model info
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            cpu_info = result.stdout.strip()

            # Assume Metal is available on all Apple Silicon Macs
            results.append(
                dict(
                    hardware_type="gpu",
                    gpu_type="Integrated (Apple Silicon)",
                    gpu_name="Apple Metal GPU",
                    driver_ver="Metal",
                    manufacturer="Apple",
                    dac_type="Metal",
                    pnp_device_id="apple-silicon",
                    id=0,
                    vram_total=None,  # Apple Silicon uses unified memory
                    vram_used=None,
                    vram_free=None,
                )
            )
            # Debug
            print(f"GPU Name: Apple Metal GPU", flush=True)
            print(f"CPU: {cpu_info}", flush=True)
            print(f"Driver: Metal (Built-in)", flush=True)
            print(f"Type: Integrated (Apple Silicon)", flush=True)
        except Exception as e:
            print(f"[UPDATER] Error detecting macOS GPU: {e}", flush=True)
            # Fallback - still assume Metal is available
            results.append(
                dict(
                    hardware_type="gpu",
                    gpu_type="Integrated",
                    gpu_name="Metal GPU",
                    driver_ver="Metal",
                    manufacturer="Apple",
                    dac_type="Metal",
                    pnp_device_id="unknown",
                    id=0,
                    vram_total=None,
                    vram_used=None,
                    vram_free=None,
                )
            )
        return results

    # Windows - Use WMI for GPU detection
    if platform.system() == "Windows":
        w = wmi.WMI()
        gpus = w.Win32_VideoController()
        # Get details
        for gpu in gpus:
            gpu_name = gpu.Name
            driver_ver = gpu.DriverVersion
            manufacturer = gpu.AdapterCompatibility
            dac_type = gpu.AdapterDACType
            pnp_device_id = gpu.PNPDeviceID
            # Heuristic to determine discrete vs integrated GPU
            if (
                "Intel" in manufacturer
                or "UMA" in dac_type
                or "VEN_8086" in pnp_device_id
            ):
                gpu_type = "Integrated (Onboard)"
            else:
                gpu_type = "Discrete (Dedicated)"
            results.append(
                dict(
                    hardware_type="gpu",
                    gpu_type=gpu_type,
                    gpu_name=gpu_name,
                    driver_ver=driver_ver,
                    manufacturer=manufacturer,
                    dac_type=dac_type,
                    pnp_device_id=pnp_device_id,
                )
            )
            # Debug
            print(f"GPU Name: {gpu_name}", flush=True)
            print(f"Driver Version: {driver_ver}", flush=True)
            print(f"Manufacturer: {manufacturer}", flush=True)
            print(f"Type: {gpu_type}", flush=True)
        # Get extra details
        gpus = GPUtil.getGPUs()
        for index, gpu in enumerate(gpus):
            results[index]["id"] = gpu.id
            results[index]["vram_total"] = gpu.memoryTotal
            results[index]["vram_used"] = gpu.memoryUsed
            results[index]["vram_free"] = gpu.memoryFree
            # Debug
            print(f"GPU ID: {gpu.id}")
            print(f"Total VRAM: {gpu.memoryTotal}MB")
            print(f"Used VRAM: {gpu.memoryUsed}MB")
            print(f"Free VRAM: {gpu.memoryFree}MB")

    return results


def download_and_extract(
    repo: str,
    tag: str,
    asset_name: str,
    target_path: str,
    files_to_extract: List[str] = None,
):
    """
    Downloads and extracts specific files from a ZIP file from a GitHub repo.

    Args:
        repo: GitHub repo in format "owner/repo"
        tag: Release tag
        asset_name: Name of the asset file
        target_path: Where to extract files
        files_to_extract: List of specific files to extract. If None, extracts all files.
    """
    url = f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"

    try:
        print(f"[UPDATER] Downloading {asset_name} from {url}...to {target_path}")

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        if response.status_code == 200:
            # Create the extraction directory if it doesn't exist
            os.makedirs(target_path, exist_ok=True)
            with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
                if files_to_extract:
                    # Debug: List all files in the archive
                    all_files = zip_file.namelist()
                    print(f"[UPDATER] Archive contains {len(all_files)} files")

                    # Only extract specified files
                    for file_name in files_to_extract:
                        try:
                            # Look for the file in the archive (may be in subdirectory)
                            matching_files = [
                                f
                                for f in all_files
                                if f.endswith(file_name) and not f.endswith("/")
                            ]

                            if matching_files:
                                for match in matching_files:
                                    # Extract file and flatten directory structure
                                    source = zip_file.open(match)
                                    target_file = os.path.join(
                                        target_path, os.path.basename(match)
                                    )
                                    with open(target_file, "wb") as target:
                                        target.write(source.read())
                                    source.close()
                                    print(
                                        f"[UPDATER] Extracted {match} -> {os.path.basename(match)}"
                                    )
                            else:
                                print(
                                    f"[UPDATER] Warning: {file_name} not found in archive"
                                )
                        except Exception as e:
                            print(f"[UPDATER] Error extracting {file_name}: {e}")
                else:
                    # Extract all files (fallback behavior)
                    zip_file.extractall(target_path)
            print(f"[UPDATER] Extraction complete.")
        else:
            print(
                f"[UPDATER] Failed to download file. Status code: {response.status_code}"
            )

        # Debug: Check results
        # for dirpath, dirnames, filenames in os.walk(target_path):
        #     print(filenames)
        #     break
    except requests.exceptions.RequestException as e:
        print(f"[UPDATER] Error downloading file: {e}")
    except Exception as e:
        print(f"[UPDATER] An error occurred: {e}")


# Install NLTK and download stopwords (req by llama-index)
# def download_extra_deps():
#     try:
#         nltk.data.find("corpora/stopwords")
#     except LookupError:
#         nltk.download("stopwords")


def check_llama_cpp_exists(file_paths):
    """
    Check if all required llama.cpp files exist.

    Args:
        file_paths: Single file path (str) or list of file paths to check

    Returns:
        True if all files exist, False otherwise
    """
    try:
        # Handle both single file and list of files
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # Check if all files exist
        for file_path in file_paths:
            if not os.path.isfile(file_path):
                print(f"[UPDATER] Missing required file: {file_path}", flush=True)
                return False

        return True
    except Exception as e:
        print(f"[UPDATER] Error checking file existence: {e}")
        return False


def install_llama_cpp(gpu: dict, tag: str, target_path: str):
    repo = "ggml-org/llama.cpp"
    is_nvidia_gpu = "NVIDIA" in gpu.get("gpu_name") or "NVIDIA" in gpu.get(
        "manufacturer"
    )

    # For macOS - Apple Silicon with Metal support
    # Set `--n-gpu-layers 999` or `-ngl -1` to use gpu acceleration
    # @TODO - Only implements support for Q4_0 (check most recent release for better support)
    if platform.system() == "Darwin":
        # Extract llama-cli binary and other files required for GPU acceleration
        files_to_extract = [
            "llama-cli",  # Main CLI (also used for embeddings with --embedding flag)
            # "llama-embedding",  # Embedding binary for GGUF models - This will be compiled and bundled
            "ggml-metal.metal",  # Metal shader source (loaded at runtime)
            "ggml-common.h",  # Common headers (required by Metal shader)
            "ggml-metal-impl.h",  # Metal implementation header
            "libggml-base.dylib",  # Base GGML library
            "libggml-blas.dylib",  # BLAS support
            "libggml-cpu.dylib",  # CPU support
            "libggml-metal.dylib",  # Metal GPU support
            "libggml-rpc.dylib",  # RPC support
            "libggml.dylib",  # Core GGML library
            "libllama.dylib",  # Main llama library (REQUIRED)
            "libggml-base.0.dylib",  # Base GGML library
            "libggml-blas.0.dylib",  # BLAS support
            "libggml-cpu.0.dylib",  # CPU support
            "libggml-metal.0.dylib",  # Metal GPU support
            "libggml-rpc.0.dylib",  # RPC support
            "libggml.0.dylib",  # Core GGML library
            "libllama.0.dylib",  # Main llama library (REQUIRED)
        ]

        # Download llama.cpp binaries for Apple Silicon (ARM64 with Metal)
        download_and_extract(
            repo=repo,
            tag=tag,
            asset_name=f"llama-{tag}-bin-macos-arm64.zip",
            target_path=target_path,
            files_to_extract=files_to_extract,
        )
        # Make the binaries executable on Unix-like systems
        for binary_name in ["llama-cli"]:
            binary_path = os.path.join(target_path, binary_name)
            if os.path.exists(binary_path):
                os.chmod(binary_path, 0o755)  # rwxr-xr-x permissions
                print(f"[UPDATER] Made {binary_path} executable", flush=True)
        print(
            "[UPDATER] Downloaded llama.cpp with Metal GPU support for macOS",
            flush=True,
        )

    # For Windows - Nvidia
    elif platform.system() == "Windows":
        if is_nvidia_gpu:
            # Extract llama-cli binaries
            llama_files = ["llama-cli.exe"]

            # Download llama.cpp binaries
            download_and_extract(
                repo=repo,
                tag=tag,
                asset_name=f"llama-{tag}-bin-win-cuda-cu12.4-x64.zip",
                target_path=target_path,
                files_to_extract=llama_files,
            )

            # Required CUDA DLLs for llama-cli to run
            cuda_dlls = [
                "cublas64_12.dll",
                "cublasLt64_12.dll",
                "cudart64_12.dll",
            ]

            # Download cuda dll's
            download_and_extract(
                repo=repo,
                tag=tag,
                asset_name="cudart-llama-bin-win-cu12.4-x64.zip",
                target_path=target_path,
                files_to_extract=cuda_dlls,
            )
        # @TODO Handle for amd gpu's
        # ...
    # @TODO Handle for Linux
    # ...


class Updater:
    def __init__(self):
        self.status = "idle"
        self.package_json = dict()
        try:
            self.package_json = get_package_json()
        except Exception as error:
            print(f"[UPDATER] Failed to read package file: {error}", flush=True)

        print(f"[UPDATER] Starting updater...", flush=True)

    def check_version(self):
        try:
            return self.package_json.get("version")
        except Exception as error:
            print(f"[UPDATER] Failed to get package version: {error}", flush=True)
            return ""

    def check_if_update(self, latest_version):
        # Check for updated launcher, ask user for download.
        curr_ver = self.package_json.get("version")
        print(
            f"[UPDATER] Checking for latest app version, current: {curr_ver} | latest: {latest_version} ...",
            flush=True,
        )
        if latest_version and curr_ver and f"v{curr_ver}" != latest_version:
            # new ver exists
            return True
        return False

    def download(self):
        """
        Note: llama.cpp binaries are now compiled and bundled during the build process.
        This method is kept for backward compatibility but no longer downloads binaries.
        """
        print("[UPDATER] Skipping binary download - binaries are pre-bundled.", flush=True)

        self.status = "progress"

        # Evaluate hardware (still useful for system info)
        gpus = get_gpu_details()

        # Check that bundled binaries exist
        deps_path = dep_path()
        target_path = os.path.join(deps_path, "servers", "llama.cpp")

        # Build list of required files based on platform
        required_files = []
        if platform.system() == "Darwin":
            # macOS - llama-cli and llama-embedding binaries
            required_files = [
                os.path.join(target_path, "llama-cli"),
                os.path.join(target_path, "llama-embedding"),
            ]
        elif platform.system() == "Windows":
            # Windows - llama-cli.exe, llama-embedding.exe + CUDA DLLs
            required_files = [
                os.path.join(target_path, "llama-cli.exe"),
                os.path.join(target_path, "llama-embedding.exe"),
                os.path.join(target_path, "cublas64_12.dll"),
                os.path.join(target_path, "cublasLt64_12.dll"),
                os.path.join(target_path, "cudart64_12.dll"),
            ]

        print("[UPDATER] Checking for bundled binaries...", flush=True)
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"[UPDATER] Warning: Missing bundled files: {missing_files}", flush=True)
            print("[UPDATER] The application may not work correctly.", flush=True)
        else:
            print("[UPDATER] All bundled binaries found.", flush=True)

        # Finished
        self.status = "complete"
        print("[UPDATER] Finished.", flush=True)

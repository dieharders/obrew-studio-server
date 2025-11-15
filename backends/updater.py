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
                check=True
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
                    vram_total="Shared",  # Apple Silicon uses unified memory
                    vram_used="N/A",
                    vram_free="N/A",
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
                    vram_total="Shared",
                    vram_used="N/A",
                    vram_free="N/A",
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
            if "Intel" in manufacturer or "UMA" in dac_type or "VEN_8086" in pnp_device_id:
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


def download_and_extract(repo: str, tag: str, asset_name: str, target_path: str):
    """
    Downloads and extracts a ZIP file from a GitHub repo.
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
                # @TODO Only extract the binary you need and delete the rest.
                zip_file.extractall(target_path)
            print(f"[UPDATER] Extracted {asset_name} to the current directory.")
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


def check_llama_cpp_exists(file_path):
    try:
        # Construct the absolute path to the file
        return os.path.isfile(file_path)
    except Exception as e:
        print(f"[UPDATER] Error checking file existence: {e}")
        return False


def install_llama_cpp(gpu: dict, tag: str, target_path: str):
    repo = "ggml-org/llama.cpp"
    is_nvidia_gpu = "NVIDIA" in gpu.get("gpu_name") or "NVIDIA" in gpu.get(
        "manufacturer"
    )

    # For macOS - Apple Silicon with Metal support
    if platform.system() == "Darwin":
        # Download llama.cpp binaries for Apple Silicon (ARM64 with Metal)
        download_and_extract(
            repo=repo,
            tag=tag,
            asset_name=f"llama-{tag}-bin-macos-arm64.zip",
            target_path=target_path,
        )
        # Make the binary executable on Unix-like systems
        binary_path = os.path.join(target_path, "llama-cli")
        if os.path.exists(binary_path):
            os.chmod(binary_path, 0o755)  # rwxr-xr-x permissions
            print(f"[UPDATER] Made {binary_path} executable", flush=True)
        print("[UPDATER] Downloaded llama.cpp with Metal GPU support for macOS", flush=True)

    # For Windows - Nvidia
    elif platform.system() == "Windows":
        if is_nvidia_gpu:
            # Download llama.cpp binaries
            download_and_extract(
                repo=repo,
                tag=tag,
                asset_name=f"llama-{tag}-bin-win-cuda-cu12.4-x64.zip",
                target_path=target_path,
            )
            # Download cuda dll's
            download_and_extract(
                repo=repo,
                tag=tag,
                asset_name="cudart-llama-bin-win-cu12.4-x64.zip",
                target_path=target_path,
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
        print("[UPDATER] Evaluating hardware dependencies...", flush=True)

        self.status = "progress"

        # Evaluate hardware
        gpus = get_gpu_details()

        # Download llama.cpp binaries
        deps_path = dep_path()
        target_path = os.path.join(deps_path, "servers", "llama.cpp")
        # Platform-aware binary name
        binary_name = "llama-cli.exe" if platform.system() == "Windows" else "llama-cli"
        file_path = os.path.join(target_path, binary_name)
        llamacpp_tag = self.package_json.get("llamacpp_tag")
        print("[UPDATER] Checking for deps...", flush=True)
        if not check_llama_cpp_exists(file_path):
            print("[UPDATER] Downloading inference binaries ...", flush=True)
            install_llama_cpp(gpu=gpus[0], tag=llamacpp_tag, target_path=target_path)
            print("[UPDATER] Download complete.", flush=True)

        # Finished
        self.status = "complete"
        print("[UPDATER] Finished.", flush=True)

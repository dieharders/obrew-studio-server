import json
from typing import Dict, List
import requests
import zipfile
import os
import sys
import wmi
from io import BytesIO
import platform
import GPUtil

# @TODO Perhaps make a small installer app with Tauri that has a GUI that can launch headless/non etc of app and install/notify of updates. We wouldn't need multiple app shortcuts and the download link can always point to one file for all platforms.


def get_gpu_details() -> List[Dict]:
    w = wmi.WMI()
    gpus = w.Win32_VideoController()
    results = []

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
                gpu_type=gpu_type,
                gpu_name=gpu_name,
                driver_ver=driver_ver,
                manufacturer=manufacturer,
                dac_type=dac_type,
                pnp_device_id=pnp_device_id,
            )
        )

    # Get extra details
    gpus = GPUtil.getGPUs()
    for index, gpu in enumerate(gpus):
        results[index]["id"] = gpu.id
        results[index]["vram_total"] = gpu.memoryTotal
        results[index]["vram_used"] = gpu.memoryUsed
        results[index]["vram_free"] = gpu.memoryFree

    # Debug
    for gpu in results:
        print(f"GPU Name: {gpu.get("gpu_name")}", flush=True)
        print("GPU ID:", gpu.get("id"))
        print(f"Driver Version: {gpu.get("driver_ver")}", flush=True)
        print(f"Manufacturer: {gpu.get("manufacturer")}", flush=True)
        print(f"Type: {gpu.get("gpu_type")}", flush=True)
        print("Total VRAM:", gpu.get("vram_total"), "MB")
        print("Used VRAM:", gpu.get("vram_used"), "MB")
        print("Free VRAM:", gpu.get("vram_free"), "MB")

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
    # For Windows - Nvidia
    if platform.system() == "Windows":
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
    # @TODO Handle for other OS's
    # ...


# Define the application's path so we can find files
def get_path(target_dir: str):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        is_compiled = sys._MEIPASS
        base_path = os.path.join(os.getcwd(), target_dir)
    except Exception:
        base_path = os.getcwd()
    return base_path


# Read package file
def read_package_json(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            package_json: dict = json.load(f)
        return package_json
    except Exception:
        print("[UPDATER] Failed to read package.")


class Updater:
    def __init__(self):
        self.status = "idle"
        print("[UPDATER] Starting updater...", flush=True)

    def check_updates(self):
        # @TODO Check for updated launcher, ask user for download.
        # ...
        return False

    def download(self):
        print("[UPDATER] Evaluating hardware dependencies...", flush=True)

        self.status = "progress"

        # Evaluate hardware
        gpus = get_gpu_details()

        # Download llama.cpp binaries
        deps_path = get_path("_deps")
        file_path = os.path.join(deps_path, "servers", "llama.cpp", "llama-cli.exe")
        target_path = os.path.join(deps_path, "servers", "llama.cpp")
        public_path = get_path("public")
        package_path = os.path.join(public_path, "package.json")
        llamacpp_tag = read_package_json(package_path).get("llamacpp_tag")
        print("[UPDATER] Checking for deps...", flush=True)
        if not check_llama_cpp_exists(file_path):
            print("[UPDATER] Downloading inference binaries ...", flush=True)
            install_llama_cpp(gpu=gpus[0], tag=llamacpp_tag, target_path=target_path)
            print("[UPDATER] Download complete.", flush=True)

        # Finished
        self.status = "complete"
        print("[UPDATER] Finished.", flush=True)

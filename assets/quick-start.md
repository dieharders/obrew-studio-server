Testing Steps for macOS Support

# 1. Clean Setup (Fresh Start)

Navigate to project directory

`cd /Users/dieharders/Projects/obrew-studio-server`

## Create a fresh virtual environment

```bash
# min version
python3.12 -m venv .venv
```

## Activate the virtual environment

```bash
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

# 2. Run in Development Mode (Simplest Test)

Make sure you're in the virtual environment

```bash
source .venv/bin/activate
```

## Run the app in dev mode

```bash
pnpm dev
```

### What should happen (macOS):

The updater will detect your Apple Silicon Mac
It will print: GPU Name: Apple Metal GPU
It will check for llama-cli binary at ~/.obrew-studio/deps/servers/llama.cpp/llama-cli
If not found, it will download llama-b4905-bin-macos-arm64.zip
It will make the binary executable with chmod 0o755
The app should launch with a GUI window (pywebview) 3. Check the Logs
Watch for these key messages in the terminal output:
[UPDATER] Starting updater...
[UPDATER] Evaluating hardware dependencies...
GPU Name: Apple Metal GPU
CPU: Apple M1/M2/M3/M4 [your chip]
Driver: Metal (Built-in)
Type: Integrated (Apple Silicon)
[UPDATER] Checking for deps...
[UPDATER] Downloading inference binaries ...
[UPDATER] Downloading llama-b4905-bin-macos-arm64.zip from https://github.com/ggml-org/llama.cpp/releases/download/b4905/llama-b4905-bin-macos-arm64.zip...
[UPDATER] Extracted llama-b4905-bin-macos-arm64.zip to the current directory.
[UPDATER] Made /Users/[your-username]/.obrew-studio/deps/servers/llama.cpp/llama-cli executable
[UPDATER] Downloaded llama.cpp with Metal GPU support for macOS
[UPDATER] Download complete.
[UPDATER] Finished.

# 4. Verify Binary Downloaded Correctly

Check if the binary exists and is executable

`ls -la ~/.obrew-studio/deps/servers/llama.cpp/`

Should show something like:

`-rwxr-xr-x 1 username staff [size] llama-cli`

# Build the standalone app

npm run build-prod-mac

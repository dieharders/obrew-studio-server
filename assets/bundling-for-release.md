# Bundling

Bundle your code dependencies and compress into an installation wizard.

## Llama.cpp Server

API docs: [https://platform.openai.com/docs/api-reference/making-requests](https://platform.openai.com/docs/api-reference/making-requests)
Server docs: [https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)

For Windows with cuBLAS (Nvidia) support.
Download pre-built binaries [here](https://github.com/ggerganov/llama.cpp/releases):

- llama-<tag>-bin-win-cuda-cu<version>-x64 (server binaries)
- cudart-llama-bin-win-cu<version>-x64 (contains dlls if client has none installed)

The binaries and dependencies for the server will be downloaded by the installer program during user's installation and be put into the `servers/` folder.

### Windows ("cmd" terminal)

Start server:

```cmd
llama-server.exe -m ..\..\text-models\your-model.gguf -c 2048 --n-predict 30
```

Start server - Advanced (no UI, offload all layers to 1st GPU):

```cmd
llama-server.exe --model ..\..\text-models\your-model.gguf --no-webui --n-predict 30 --ctx-size 2048 --port 8090 --n-gpu-layers 100 --device CUDA0
```

Prompt the CLI:

```cmd
llama-cli.exe --model ..\..\text-models\your-model.gguf --prompt "how big is the french flag?" --ctx-size 2048 --temp 0.65 --seed 69 --n-predict 30 --n-gpu-layers 100 --device CUDA0
```

## Inno Installer Setup Wizard

This utility will take your exe and dependencies and compress the files, then wrap them in a user friendly executable that guides the user through installation.

1. Download Inno Setup from (here)[https://jrsoftware.org/isinfo.php]

2. Install and run the setup wizard for a new script

3. Follow the instructions and before it asks to compile the script, cancel and inspect the script where it points to your included files/folders

4. Be sure to append `/[your_included_folder_name]` after the `DestDir: "{app}"`. So instead of `{app}` we have `{app}/assets`. This will ensure it points to the correct paths of the added files you told pyinstaller to include.

5. After that compile the script and it should output your setup file where you specified (or project root).

# Releasing

## Create a release on Github with link to installer

1. Create a tag with:

Increase the patch version by 1 (x.x.1 to x.x.2)

```bash
yarn version --patch
```

Increase the minor version by 1 (x.1.x to x.2.x)

```bash
yarn version --minor
```

Increase the major version by 1 (1.x.x to 2.x.x)

```bash
yarn version --major
```

2. Create a new release in Github and choose the tag just created or enter a new tag name for Github to make.

3. Drag & Drop the binary file you wish to bundle with the release. Then hit done.

4. If the project is public then the latest release's binary should be available on the web to anyone with the link:

https://github.com/[github-user]/[project-name]/releases/latest/download/[installer-file-name]

[Back to main README](../README.md)

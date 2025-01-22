# Compiling

Take all dependencies, dlls, source code and bundle with an executable. Be sure to generate self-signed certs for easy SSL setup in local environment.

## Compiling Nvida CUDA toolkit deps:

If you already have the required toolkit files installed and have built for GPU then the necessary GPU drivers/dlls should be detected by PyInstaller and included in the `_deps` dir.

## Compiling with PyInstaller:

This is handled automatically by npm scripts so you do not need to execute these manually. The -F flag bundles everything into one .exe file.

To install the pyinstaller tool:

```bash
pip install -U pyinstaller
```

Then use it to bundle a python script:

```bash
pyinstaller -c -F your_program.py
```

## Compiling with auto-py-to-exe (recommended)

This is a GUI tool that greatly simplifies the process. You can also save and load configs. It uses PyInstaller under the hood and requires it to be installed. Please note if using a conda or virtual environment, be sure to install both PyInstaller and auto-py-to-exe in your virtual environment and also run them from there, otherwise one or both will build from incorrect deps.

\*_Note_, you will need to edit paths for the following in `auto-py-to-exe` to point to your base project directory:

- Settings -> Output directory
- Additional Files
- Script Location
- Icon Location

To run:

```bash
auto-py-to-exe
```

# Packaging

Compress & bundle your code into an installation wizard.

## Inno Installer Setup Wizard

This utility will take your exe and dependencies and compress the files, then wrap them in a user friendly executable that guides the user through installation.

1. Download Inno Setup from (here)[https://jrsoftware.org/isinfo.php]

2. Install and run the setup wizard for a new script

3. Follow the instructions and before it asks to compile the script, cancel and inspect the script where it points to your included files/folders

4. Be sure to append `/[your_included_folder_name]` after the `DestDir: "{app}"`. So instead of `{app}` we have `{app}/assets`. This will ensure it points to the correct paths of the added files you told pyinstaller to include.

5. After that compile the script and it should output your setup file where you specified (or project root).

[Back to main README](../README.md)

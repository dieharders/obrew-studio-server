# Building locally

Take all dependencies, dlls, source code and bundle with an executable. Be sure to generate self-signed certs for easy SSL setup in local environment.

## Build binary with PyInstaller:

This is handled automatically by npm scripts so you do not need to execute these manually. The -F flag bundles everything into one .exe file.

To install the pyinstaller tool:

```bash
pip install -U pyinstaller
```

Then use it to bundle a python script:

```bash
pyinstaller -c -F your_program.py
```

## Build binary with auto-py-to-exe (recommended)

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

# Build using Github Action

Fork this repo in order to access a manual trigger to build for each platform (Windows, MacOS, Linux) and upload a release.

Modify the `release.yml` build pipeline. Workflow permissions must be set to "Read and write". Any git tags created before a workflow existed will not be usable for that workflow. You must specify a tag to run from (not a branch name).

Initiate the Workflow Manually:

1. Navigate to the "Actions" tab in your GitHub repository.
2. Select the "Manual Release" workflow.
3. Click on "Run workflow" and provide the necessary inputs:

   - release_name: The title of the release.
   - release_notes (optional): Notes or changelog for the release.
   - release_type: ("draft", "public", "private")

[Back to main README](../README.md)

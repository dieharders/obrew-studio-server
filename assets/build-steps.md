# Build Locally

Take all dependencies, dlls, source code and bundle with an executable. Be sure to generate self-signed certs for easy SSL setup in local environment.

## Build binary with PyInstaller:

Install the pyinstaller tool:

```bash
pip install -U pyinstaller
```

It can be used to bundle a python script for example:

```bash
# The -F flag bundles everything into one .exe file.
pyinstaller -c -F your_program.py
```

This is handled automatically by the build script:

```bash
# You may need to edit the python virtual env path to match yours
yarn build:app
```

## Build binary with auto-py-to-exe (optional)

This is a GUI tool that greatly simplifies the process. You can also save and load configs. It uses PyInstaller under the hood and requires it to be installed. Please note if using a conda or virtual environment, be sure to install both pyInstaller and auto-py-to-exe in your virtual environment and also run them from there, otherwise one or both will build from incorrect deps.

Run:

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

<br>

[Back to main README](../README.md)

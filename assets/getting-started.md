# Getting Started

## Running the release executable

If you get a "Permission Denied" error, try running the executable with Admin privileges.

There are two shortcuts installed, the normal executable and one for "headless" mode. In headless mode the backend will run in the background without a GUI window. This is ideal for automation or development since you can use command line arguments to specify how you run the service:

- --host=0.0.0.0
- --port=8008
- --headless=True
- --mode=dev or prod (this enables/disables logging)

## Run/Build from source code

### Install Python dependencies

Install dependencies for python listed in requirements.txt file:

Be sure to run this command with admin privileges. This command is optional and is also run on each `yarn build`.

```bash
pip install -r requirements.txt
# or
yarn python-deps
```

### Install WebUI dependencies (for Front-End GUI)

Not strictly required, but if you intend to work with the UI files (html/css/js) and want linting, etc. then run:

```bash
yarn install
```

### Start the backend

Right-click over `backends/main.py` and choose "run python file in terminal" to start server:

Or

```bash
# from working dir
python backends/main.py
```

Or using yarn (recommended)

```bash
yarn server:dev
# or
yarn server:prod
# or to run headless (production)
yarn server:headless-prod
# or to run headless (development)
yarn server:headless-dev
```

The Obrew api server will be running on [https://localhost:8008](https://localhost:8008)

\*_Note_ if the server fails to start be sure to run `yarn makecert` command to create certificate files necessary for https (these go into `_deps/public` folder).

## Managing Python dependencies

It is highly recommended to use a package manager like Anaconda to manage Python installations and the versions of dependencies they require. This allows you to create virtual environments from which you can install different versions of software and build/deploy from within this sandboxed environment.

To update PIP package installer:

```bash
conda update pip
```

### Create/Switch between virtual environments (Conda)

The following commands should be done in `Anaconda Prompt` terminal. If on Windows, `run as Admin`.

1. Create a new environment. This project uses `3.12.3`:

```bash
conda create --name env1 python=3.12
```

2. To work in this env, activate it:

```bash
conda activate env1
```

3. When you are done using it, deactivate it:

```bash
conda deactivate
```

4. If using an IDE like VSCode, you must apply your newly created virtual environment by selecting the `python interpreter` button at the bottom when inside your project directory.

### Create/Switch between virtual environments (venv)

venv is the recommended tool for creating virtual environments for Python v3.6 and up.

1. Create a new environment:

```bash
python -m venv .virtualenvs/myenv
```

2. Activate the environment:

```cmd
.virtualenvs/myenv/Scripts/activate.bat
```

OR

```powershell
.virtualenvs/myenv/Scripts/Activate.ps1
```

OR on MacOS/Linux

```bash
source .virtualenvs/myvenv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

[Back to main README](../README.md)

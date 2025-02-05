import os
import sys
import socket
import signal
from dotenv import load_dotenv

# Custom
from ui.view import Webview
from ui.api_ui import Api
from core import common
from updater import Updater

###############
### Methods ###
###############


# Parse runtime arguments passed to script
def _parse_runtime_args():
    # Command-line arguments are accessed via sys.argv
    arguments = sys.argv[1:]
    # Initialize default variables to store parsed arguments
    mode = None
    host = "0.0.0.0"
    port = "8008"
    headless = "False"
    # Iterate through arguments and parse them
    for arg in arguments:
        if arg.startswith("--host="):
            host = arg.split("=")[1]
        if arg.startswith("--port="):
            port = arg.split("=")[1]
        if arg.startswith("--mode="):
            mode = arg.split("=")[1]
        if arg.startswith("--headless="):
            headless = arg.split("=")[1]
    return {
        "host": host,
        "port": port,
        "mode": mode,
        "headless": headless,
    }


def _get_server_info():
    # Display the local IP address of this server
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    ssl = common.get_ssl_env()
    if ssl:
        SCHEME = "https"
    else:
        SCHEME = "http"
    remote_ip = f"{SCHEME}://{IPAddr}"
    local_ip = f"{SCHEME}://localhost"
    return {
        "local_ip": local_ip,
        "remote_ip": remote_ip,
    }


# Graceful shutdown, close everything and cleanup
def _close_app(api=None):
    try:
        # Do any cleanup here...
        if api and api.api_server:
            api.api_server.shutdown()
        print(f"{common.PRNT_APP} Closing app.", flush=True)
    except:
        print("Failed to close App.", flush=True)


#############
### Start ###
#############


def main():
    try:
        # Webview api
        window_api = Api(
            port=port,
            host=host,
            is_prod=is_prod,
            is_dev=is_dev,
            is_debug=is_debug,
            webui_url=webui_url,
            get_server_info=_get_server_info,
            updater=Updater(),  # downloads deps and signals user to updates
        )

        # Handle premature keyboard interrupt
        def signal_handler(sig, frame):
            print(
                f"{common.PRNT_APP} Signal received. Main process interrupted. Shutting down.",
                flush=True,
            )
            _close_app(api=window_api)  # sys.exit(0)

        # Listen for signal handler for SIGINT (Ctrl+C, KeyboardInterrupt)
        signal.signal(signal.SIGINT, signal_handler)

        # Start server process on startup for headless mode (otherwise, we do this via webui or cli)
        if is_headless:
            config = dict(host=host, port=port)
            window_api.start_headless_server(config)

        # Show a window (non-headless mode)
        if not is_headless:
            server_info = _get_server_info()
            remote_ip = server_info["remote_ip"]
            view_instance = Webview(
                js_api=window_api,
                is_prod=is_prod,
                is_dev=is_dev,
                is_debug=is_debug,
                remote_ip=remote_ip,
                IS_WEBVIEW_SSL=False,  # always run app FE as http
                webui_url=webui_url,
            )
            view_instance.create_window()

            # Handle window closing
            def on_window_closing():
                view_instance.api_server.shutdown()

            window_handle = view_instance.webview_window
            window_handle.events.closing += on_window_closing
            # Start window
            start_ui = view_instance.callback
            start_ui()

    except Exception as e:
        print(f"{common.PRNT_APP} Main process error: {e}", flush=True)


# This script is the loader for the rest of the backend. It only handles UI and starting dependencies.
if __name__ == "__main__":
    print(f"{common.PRNT_APP} Starting app...", flush=True)

    # Path to the .env file in either the parent or /_deps directory
    env_path = common.dep_path(".env")
    load_dotenv(env_path)

    # Check what env is running - prod/dev
    build_env = _parse_runtime_args()

    # Initialize global data
    host = build_env["host"]
    port = build_env["port"]
    is_headless = build_env["headless"] == "True"  # no UI
    is_debug = hasattr(sys, "gettrace") and sys.gettrace() is not None
    is_dev = build_env["mode"] == "dev" or is_debug
    is_prod = build_env["mode"] == "prod" or not is_dev
    webui_url = "https://studio.openbrewai.com"

    # Comment out if you want to debug on prod build (or set --mode=prod flag in command)
    if is_prod:
        # Remove prints in prod when deploying in window mode
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    main()

import os
import sys
import socket
import signal
import platform
import multiprocessing
from typing import Tuple
from dotenv import load_dotenv
import ctypes

# Custom
from ui.view import Webview
from ui.api_ui import ApiUI
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
        print(f"{common.PRNT_APP} Failed to close App.", flush=True)


def _get_screen_res() -> Tuple:
    try:
        # macOS - Use AppKit to get screen resolution
        if platform.system() == "Darwin":
            from AppKit import NSScreen

            screen = NSScreen.mainScreen()
            screen_size = (
                int(screen.frame().size.width),
                int(screen.frame().size.height),
            )
            return screen_size
        # Windows - Use ctypes to get screen resolution
        elif platform.system() == "Windows":
            user32 = ctypes.windll.user32
            screen_size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            return screen_size
        else:
            # Linux or other platforms
            print(
                f"{common.PRNT_APP} Screen resolution detection not implemented for this platform.",
                flush=True,
            )
            return None
    except Exception as e:
        print(
            f"{common.PRNT_APP} Failed to get the current screen resolution: {e}",
            flush=True,
        )
        return None


#############
### Start ###
#############


def main():
    try:
        # Webview api
        window_api = ApiUI(
            port=port,
            host=host,
            is_prod=is_prod,
            is_dev=is_dev,
            is_debug=is_debug,
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
            # Keep main thread alive so daemon server thread doesn't exit
            if window_api.api_server and window_api.api_server.server_thread:
                window_api.api_server.server_thread.join()

        # Show a window (GUI mode)
        if not is_headless:
            screen_res = _get_screen_res()
            server_info = _get_server_info()
            remote_ip = server_info["remote_ip"]
            view_instance = Webview(
                js_api=window_api,
                is_prod=is_prod,
                is_dev=is_dev,
                is_debug=is_debug,
                remote_ip=remote_ip,
                IS_WEBVIEW_SSL=False,  # always run app FE as http
                screen_size=screen_res,
            )
            view_instance.create_window()

            # Handle window closing
            def on_window_closing():
                print(
                    f"{common.PRNT_APP} Window closing, shutting down server...",
                    flush=True,
                )
                if view_instance.api_server:
                    view_instance.api_server.shutdown()
                # Give server time to cleanup
                import time

                time.sleep(0.5)
                return True

            window_handle = view_instance.webview_window
            window_handle.events.closing += on_window_closing
            # Start window
            start_ui = view_instance.callback
            start_ui()

    except Exception as e:
        print(f"{common.PRNT_APP} Main process error: {e}", flush=True)


# This script is the loader for the rest of the backend. It only handles UI and starting dependencies.
if __name__ == "__main__":
    # Required for PyInstaller on Windows/macOS to prevent spawning duplicate windows
    # When subprocess operations happen (like downloads), Python may spawn a new process that re-executes the main script, creating a second window.
    multiprocessing.freeze_support()

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

    # Comment out if you want to debug on prod build (or set --mode=prod flag in command)
    if is_prod:
        # Remove prints in prod when deploying in window mode
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    main()

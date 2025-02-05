# import tkinter as tk
# from tkinter import ttk
# import threading
import os
import sys
from typing import Type
import webview
from webview.errors import JavascriptException
from api_server import ApiServer
from ui.api_ui import Api
from core import common


# def _run_app_window():
#     # Start the API server in a separate thread from main
#     window_thread = threading.Thread(target=GUI)
#     window_thread.daemon = True  # let the parent kill the child thread at exit
#     window_thread.start()
#     return window_thread

# Create and run the Tkinter window
# def GUI(menu_api):
#     color_bg = "#333333"
#     root = tk.Tk()
#     root.title("Obrew Server")
#     root.geometry("500x500")
#     # Since /public folder is bundled inside _deps, we need to read from root `sys._MEIPASS`
#     root.iconbitmap(default=common.dep_path(os.path.join("public", "favicon.ico")))
#     root.configure(bg=color_bg)
#     # Render title
#     Font_tuple = ("Verdana", 64, "bold")
#     root.bind("<Escape>", lambda e: e.widget.quit())
#     tk.Label(root, text="O🍺brew", font=Font_tuple).pack(fill=tk.BOTH, expand=True)
#     # Render button for connection page
#     style = ttk.Style()
#     style.configure(
#         "TButton",
#         font=("Verdana", 14),
#         borderwidth=0,
#         padding=10,
#         background="grey",
#         foreground="black",
#     )
#     style.map(
#         "TButton",
#         background=[("pressed", "black"), ("active", "grey")],
#         foreground=[("pressed", "grey"), ("active", "black")],
#     )
#     button = ttk.Button(
#         root, text="Start Here", command=menu_api.open_browser, style="TButton"
#     )
#     button.pack(pady=20)
#     # Run UI
#     root.mainloop()


# WebView window
def Webview(
    js_api: Type[Api],
    is_prod: bool,
    is_dev: bool,
    is_debug: bool,
    remote_ip: str,
    webui_url: str,
    IS_WEBVIEW_SSL: bool,
):
    try:
        # Production html files will be put in `_deps/public` folder
        base_path = sys._MEIPASS
        ui_path = os.path.join(base_path, "public", "index.html")
    except Exception:
        ui_path = "ui/public/index.html"

    webview_window = webview.create_window(
        title="Obrew Studio",
        url=ui_path,
        js_api=js_api,
        width=720,
        height=1050,
        min_size=(350, 450),
        fullscreen=False,
        # http_port=3000,
        # draggable=True,
        # transparent=True,
        # frameless=True,
        # easy_drag=True,
    )

    # Start the window
    def callback():
        webview.start(ssl=IS_WEBVIEW_SSL, debug=is_dev)

    # Set window to fullscreen
    def toggle_fullscreen():
        webview_window.toggle_fullscreen()

    # Tells front-end javascript to navigate to the webui
    def launch_webui():
        try:
            if not webview_window:
                raise Exception("Window is not initialized yet.")
            # Invoke function from the javascript context
            webview_window.evaluate_js("launchWebUI()")
            return ""
        except JavascriptException as e:
            print(f"{common.PRNT_APP} Javascript exception occured: {e}")
        except Exception as e:
            print(f"{common.PRNT_APP} Failed to launch WebUI: {e}")

    def launch_webui_failed():
        try:
            if not webview_window:
                raise Exception("Window is not initialized yet.")
            webview_window.evaluate_js("launchWebUIFailed()")
            return ""
        except Exception as e:
            print(f"{common.PRNT_APP} Failed to callback launch WebUI: {e}")

    # Start the API server. Only used for window mode.
    def start_server(config):
        try:
            print(f"{common.PRNT_APP} Starting API server...", flush=True)
            api_server = ApiServer(
                is_prod=is_prod,
                is_dev=is_dev,
                is_debug=is_debug,
                remote_url=remote_ip,
                SERVER_HOST=config["host"],
                SERVER_PORT=int(config["port"]),
                selected_webui_url=config.get("webui"),
                hosted_webui_url=webui_url,
                on_startup_callback=launch_webui,
            )
            api_server.startup()
            return api_server
        except Exception as e:
            print(f"{common.PRNT_APP} Failed to start API server. {e}", flush=True)
            launch_webui_failed()

    # Expose an inline func before runtime
    webview_window.expose(launch_webui)
    webview_window.expose(start_server)
    webview_window.expose(launch_webui_failed)
    webview_window.expose(toggle_fullscreen)

    return dict(handle=webview_window, callback=callback)

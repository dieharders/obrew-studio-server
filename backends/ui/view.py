# import tkinter as tk
# from tkinter import ttk
# import threading
import os
import sys
from typing import Tuple, Type
import webview
from webview.errors import JavascriptException
from api_server import ApiServer
from ui.api_ui import ApiUI
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


class Webview:
    def __init__(
        self,
        js_api: Type[ApiUI],
        is_prod: bool,
        is_dev: bool,
        is_debug: bool,
        remote_ip: str,
        IS_WEBVIEW_SSL: bool,
        screen_size: Tuple | None,
    ):
        self.webview_window = None
        self.callback = None
        self.api_server: ApiServer | None = None
        self.js_api = js_api
        self.is_prod = is_prod
        self.is_dev = is_dev
        self.is_debug = is_debug
        self.remote_ip = remote_ip
        self.IS_WEBVIEW_SSL = IS_WEBVIEW_SSL
        self.screen_size = screen_size

    # WebView window
    def create_window(self):
        try:
            # Production html files will be put in `_deps/public` folder
            base_path = sys._MEIPASS
            ui_path = os.path.join(base_path, "public", "index.html")
        except Exception:
            ui_path = "ui/public/index.html"

        # Calc window size (square aspect ratio)
        screen_x = 640
        screen_y = 640
        if self.screen_size and self.screen_size[1] != 0:
            # Set to fill entire screen
            # screen_x = self.screen_size[0]
            # screen_y = self.screen_size[1]
            #
            # Set to fraction of screen size (square aspect)
            screen_y = int(self.screen_size[1] // 1.5)
            screen_x = int(self.screen_size[0] // 1.5)

        self.webview_window = webview.create_window(
            title="Obrew Studio",
            url=ui_path,
            js_api=self.js_api,
            width=screen_x,
            height=screen_y,
            min_size=(300, 300),
            fullscreen=False,
            # http_port=3000,
            # draggable=True,
            # transparent=True,
            # frameless=True,
            # easy_drag=True,
        )

        # A hook to start the window
        def callback():
            webview.start(ssl=self.IS_WEBVIEW_SSL, debug=self.is_dev)

        self.callback = callback

        # Set window to fullscreen
        def toggle_fullscreen():
            self.webview_window.toggle_fullscreen()

        # Tells front-end javascript to navigate to the webui
        def launch_webui():
            try:
                if not self.webview_window:
                    raise Exception("Window is not initialized yet.")
                # Invoke function from the javascript context
                self.webview_window.evaluate_js("launchWebUI()")
                return ""
            except JavascriptException as e:
                print(f"{common.PRNT_APP} Javascript exception occured: {e}")
            except Exception as e:
                print(f"{common.PRNT_APP} Failed to launch WebUI: {e}")

        def launch_webui_failed(e: str):
            try:
                if not self.webview_window:
                    raise Exception("Window is not initialized yet.")
                msg = repr(e)
                self.webview_window.evaluate_js(f"launchWebUIFailed({msg})")
                return ""
            except Exception as e:
                print(f"{common.PRNT_APP} Failed to callback launch WebUI: {e}")

        # Start the API server. Only used for window mode.
        def start_server(config: dict):
            try:
                print(f"{common.PRNT_APP} Starting API server...", flush=True)
                self.api_server = ApiServer(
                    is_prod=self.is_prod,
                    is_dev=self.is_dev,
                    is_debug=self.is_debug,
                    remote_url=self.remote_ip,
                    SERVER_HOST=config.get("host"),
                    SERVER_PORT=int(config.get("port")),
                    selected_webui_url=config.get("webui"),
                    on_startup_callback=launch_webui,
                )
                self.api_server.startup()
            except Exception as e:
                print(f"{common.PRNT_APP} Failed to start API server. {e}", flush=True)
                launch_webui_failed(str(e))

        # Expose an inline func before runtime
        self.webview_window.expose(launch_webui)
        self.webview_window.expose(start_server)
        self.webview_window.expose(launch_webui_failed)
        self.webview_window.expose(toggle_fullscreen)

import os
import pyqrcode
import webbrowser
from core import common
from typing import Type
from api_server import ApiServer
from updater import Updater
from dotenv import set_key


# Inject these Python funcs into javascript context. Funcs must be sync.
class ApiUI:
    def __init__(
        self,
        port,
        host,
        is_prod,
        is_dev,
        is_debug,
        get_server_info,
        updater: Type[Updater],
    ):
        self.api_server = None
        self.is_prod = is_prod
        self.is_dev = is_dev
        self.is_debug = is_debug
        self.port = port
        self.host = host
        package_json = common.get_package_json()
        self.hosted_apps = package_json.get("hosted_apps_urls", [])
        self.get_server_info = get_server_info
        self.updater = updater

    # Save .env vals and other pre-launch settings
    def save_settings(self, settings: dict):
        try:
            # Get path to .env file in project root
            env_path = common.app_path(".env")

            # Create .env file if it doesn't exist
            if not os.path.exists(env_path):
                with open(env_path, "w") as f:
                    f.write("# Obrew Server Environment Variables\n")

            # Save .env values to file and update environment
            for key, value in settings.items():
                if key:  # Allow empty values to clear settings
                    clean_value = value.strip().replace(" ", "") if value else ""
                    # Write to .env file without quotes
                    set_key(env_path, key, clean_value, quote_mode="never")
                    # Also update in-memory environment
                    os.environ[key] = clean_value

            print(f"{common.PRNT_APP} Settings saved to .env file", flush=True)
            return
        except Exception as e:
            print(
                f"{common.PRNT_APP} Failed to update .env values: {e}",
                flush=True,
            )

    def get_ssl_setting(self):
        """Get the current SSL setting"""
        return common.get_ssl_env()

    def update_settings_page(self):
        try:
            # Read in .env vals
            cors = os.getenv("CUSTOM_ORIGINS", "")
            adminWhitelist = os.getenv("WHITELIST_ADMIN_IP", "")
            llamaIndexAPIKey = os.getenv("LLAMA_CLOUD_API_KEY", "")
            page_data = dict(
                ssl=common.get_ssl_env(),
                cors=cors,
                adminWhitelist=adminWhitelist,
                llamaIndexAPIKey=llamaIndexAPIKey,
            )
            return page_data
        except Exception as e:
            print(
                f"{common.PRNT_APP} Failed to update 'Main' page: {e}",
                flush=True,
            )

    def update_entry_page(self):
        try:
            # Download deps on UI startup
            if self.updater.status == "idle":
                self.updater.download()
            page_data = dict()
            return page_data
        except Exception as e:
            print(
                f"{common.PRNT_APP} Failed to update Entry page: {e}",
                flush=True,
            )

    # Get current version of app
    def get_current_version(self):
        try:
            return self.updater.check_version() or ""
        except Exception as e:
            print(
                f"{common.PRNT_APP} Failed to fetch current app version: {e}",
                flush=True,
            )
            return ""

    # Check for latest version of the app engine
    def check_is_latest_version(self, latest_version):
        try:
            new_ver_exists = self.updater.check_if_update(latest_version=latest_version)
            return new_ver_exists
        except Exception as e:
            print(
                f"{common.PRNT_APP} Failed to fetch latest release: {e}",
                flush=True,
            )
            return False

    # Generate links for user to connect an external device to this machine's
    # locally running server instance.
    # QRcode generation -> https://github.com/arjones/qr-generator/tree/main
    def update_main_page(self, port: str, selected_webui_url: str):
        try:
            PORT = port or self.port
            server_info = self.get_server_info()
            # remote_url already includes protocol (e.g., "http://192.168.2.86")
            remote_url = server_info["remote_ip"]
            local_url = server_info["local_ip"]
            server_url = f"{remote_url}:{PORT}"

            # Generate QR code for each hosted app
            hosted_apps_with_qr = []
            for app in self.hosted_apps:
                app_copy = app.copy()
                app_url = app.get("url")
                if app_url:
                    app_qr_url = f"{app_url}?serverUrl={server_url}"
                    qr_code = pyqrcode.create(app_qr_url)
                    app_copy["qr_data"] = qr_code.png_as_base64_str(scale=5)
                    app_copy["qr_url"] = app_qr_url
                hosted_apps_with_qr.append(app_copy)

            page_data = dict(
                local_url=local_url,
                remote_url=remote_url,
                host=self.host,
                port=PORT,
                hosted_apps=hosted_apps_with_qr,
            )
            return page_data
        except Exception as e:
            print(
                f"{common.PRNT_APP} Failed to update Main page: {e}",
                flush=True,
            )

    # Start the API server
    def start_headless_server(self, config):
        try:
            # Download deps on server startup
            if self.updater.status == "idle":
                self.updater.download()

            server_info = self.get_server_info()
            remote_ip = server_info["remote_ip"]
            print(f"{common.PRNT_APP} Starting headless API server...", flush=True)
            self.api_server = ApiServer(
                is_prod=self.is_prod,
                is_dev=self.is_dev,
                is_debug=self.is_debug,
                SSL_ENABLED=common.get_ssl_env(),
                remote_url=remote_ip,
                SERVER_HOST=config["host"],
                SERVER_PORT=int(config["port"]),
            )
            self.api_server.startup()
            return
        except Exception as e:
            print(f"{common.PRNT_APP} Failed to start API server. {e}", flush=True)
            raise  # Re-raise so JavaScript receives the error

    # def start_server_process(self, config):
    #     process = Process(target=self.start_server, args=[config])
    #     self.server_process = process
    #     app.state.server_process = process  # provide to outside context
    #     process.daemon = True
    #     process.start()

    # Send shutdown server request
    def shutdown_server(self, *args):
        try:
            if self.api_server:
                # Does not exist in GUI mode since api not set in this class
                self.api_server.shutdown()
            print(f"{common.PRNT_APP} Shutting down server.", flush=True)
            return
        except Exception as e:
            print(
                f"{common.PRNT_APP} Error, server forced to shutdown: {e}",
                flush=True,
            )

    # Open URL in system's default browser
    def open_url_in_browser(self, url: str):
        try:
            webbrowser.open(url)
            print(f"{common.PRNT_APP} Opening URL in browser: {url}", flush=True)
            return
        except Exception as e:
            print(
                f"{common.PRNT_APP} Failed to open URL in browser: {e}",
                flush=True,
            )

import os
import signal
import sys
import json
import uvicorn
import asyncio
import httpx
from collections.abc import Callable
from fastapi import (
    FastAPI,
    APIRouter,
)
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Custom
from embeddings.vector_storage import Vector_Storage
from core import common, classes
from services.route import router as services
from embeddings.route import router as embeddings
from inference.route import router as text_inference
from storage.route import router as storage


class ApiServer:
    def __init__(
        self,
        is_prod: bool,
        is_dev: bool,
        is_debug: bool,
        remote_url: str,
        SERVER_HOST: str,
        SERVER_PORT: int,
        selected_webui_url: str = "",
        SSL_ENABLED: bool | None = None,
        on_startup_callback: Callable | None = None,
    ):
        try:
            # Init logic here
            self.remote_url = remote_url
            self.SERVER_HOST = SERVER_HOST or "0.0.0.0"
            self.SERVER_PORT = SERVER_PORT or 8008
            self.SSL_ENABLED = SSL_ENABLED or common.get_ssl_env()
            if self.SSL_ENABLED:
                self.XHR_PROTOCOL = "https"
            else:
                self.XHR_PROTOCOL = "http"
            self.is_prod = is_prod
            self.is_dev = is_dev
            self.is_debug = is_debug
            self.selected_webui_url = selected_webui_url
            self.on_startup_callback = on_startup_callback
            # Get version from package file
            package_json = common.get_package_json()
            self.api_version = package_json.get("version")
            # Comment out if you want to debug on prod build
            if self.is_prod:
                # Remove prints in prod when deploying in window mode
                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")

            # Get paths for SSL certificate
            self.SSL_KEY: str = common.dep_path(os.path.join("public", "key.pem"))
            self.SSL_CERT: str = common.dep_path(os.path.join("public", "cert.pem"))
            # Configure CORS settings
            self.CUSTOM_ORIGINS_ENV: str = os.getenv("CUSTOM_ORIGINS")
            CUSTOM_ORIGINS = (
                self.CUSTOM_ORIGINS_ENV.split(",") if self.CUSTOM_ORIGINS_ENV else []
            )
            self.origins = [
                # "https://hoppscotch.io",  # (optional) for testing endpoints
                # "https://studio.openbrewai.com",  # official webapp frontend address
                self.selected_webui_url,  # (required) client app origin (user selected from menu)
                *CUSTOM_ORIGINS,
                # "*",  # or allow all
            ]
            # Start server
            self.app = self._create_app()
        except (Exception, FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"{common.PRNT_API} An unexpected error occurred: {e}")

    ###############
    ### Methods ###
    ###############

    def _create_app(self) -> classes.FastAPIApp:
        @asynccontextmanager
        async def lifespan(app: classes.FastAPIApp):
            print(f"{common.PRNT_API} Lifespan startup", flush=True)
            # Initialize global data here
            app.state.api = self
            app.state.request_queue = asyncio.Queue()
            app.state.db_client = None
            app.state.llm = None  # Set each time user loads a model
            # https://www.python-httpx.org/quickstart/
            app.state.requests_client = httpx.Client()

            # Tell front-end to go to webui
            if self.on_startup_callback:
                self.on_startup_callback()

            yield
            # Do shutdown cleanup here...
            print(f"{common.PRNT_API} Lifespan shutdown", flush=True)

        # Create FastAPI instance
        app_inst = FastAPI(
            title="Obrew Studio Server", version=self.api_version, lifespan=lifespan
        )

        # Add CORS support
        app_inst.add_middleware(
            CORSMiddleware,
            allow_origins=self.origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add routes
        self._add_routes(app_inst)
        return app_inst

    def shutdown(self, *args):
        try:
            print(f"{common.PRNT_API} Server forced to shutdown.", flush=True)
            if self.app.state.llm:
                self.app.state.llm.unload()
                os.kill(os.getpid(), signal.SIGTERM)  # or SIGINT
        except Exception as e:
            print(
                f"{common.PRNT_API} Failed to shutdown API server. Error: {e}",
                flush=True,
            )

    def startup(self):
        try:
            print(
                f"{common.PRNT_API} Refer to API docs:\n-> {self.XHR_PROTOCOL}://localhost:{self.SERVER_PORT}/docs \nOR\n-> {self.remote_url}:{self.SERVER_PORT}/docs",
                flush=True,
            )
            errMsg = "Server is already running on specified port. Please choose an available free port or close the duplicate app."
            # Start the ASGI server (https)
            if self.SSL_ENABLED:
                print(f"{common.PRNT_API} API server starting with SSL...", flush=True)
                if common.check_open_port(self.SERVER_PORT) != 0:
                    uvicorn.run(
                        self.app,
                        host=self.SERVER_HOST,
                        port=self.SERVER_PORT,
                        log_level="info",
                        # Include these to host over https. If server fails to start make sure the .pem files are generated in _deps/public dir
                        ssl_keyfile=self.SSL_KEY,
                        ssl_certfile=self.SSL_CERT,
                    )
                else:
                    print(f"{common.PRNT_API} {errMsg}", flush=True)
                    raise Exception(errMsg)
            # Start the ASGI server (http)
            else:
                print(f"{common.PRNT_API} API server starting...", flush=True)
                if common.check_open_port(self.SERVER_PORT) != 0:
                    uvicorn.run(
                        self.app,
                        host=self.SERVER_HOST,
                        port=self.SERVER_PORT,
                        log_level="info",
                    )
                else:
                    print(f"{common.PRNT_API} {errMsg}", flush=True)
                    raise Exception(errMsg)
        except KeyboardInterrupt as e:
            print(
                f"{common.PRNT_API} API server ended by Keyboard interrupt. {e}",
                flush=True,
            )
        except Exception as e:
            print(f"{common.PRNT_API} API server shutdown. Error: {e}", flush=True)
            raise Exception(f"Error: {e}")

    # Expose the FastAPI instance
    def get_app(self) -> FastAPI:
        """Expose the FastAPI app instance."""
        return self.app

    ##############
    ### Routes ###
    ##############

    def _add_routes(self, app: FastAPI):
        # Redirect requests to our custom endpoints
        # from fastapi import Request
        # @app.middleware("http")
        # async def redirect_middleware(request: Request, call_next):
        #     return await redirects.text(request, call_next, str(app.PORT_TEXT_INFERENCE))

        # Import routes
        endpoint_router = APIRouter()
        endpoint_router.include_router(
            services, prefix="/v1/services", tags=["services"]
        )
        endpoint_router.include_router(
            embeddings, prefix="/v1/memory", tags=["embeddings"]
        )
        endpoint_router.include_router(storage, prefix="/v1/persist", tags=["storage"])
        endpoint_router.include_router(
            text_inference, prefix="/v1/text", tags=["text inference"]
        )
        app.include_router(endpoint_router)

        # Keep server/database alive. Not currently used.
        # @app.get("/v1/ping")
        # def ping() -> classes.PingResponse:
        #     try:
        #         vector_storage = Vector_Storage(app=self.app)
        #         db = vector_storage.db_client
        #         # @TODO Perform this inside the Vector_Storage class
        #         db.heartbeat()
        #         return {"success": True, "message": "pong"}
        #     except Exception as e:
        #         print(f"{common.PRNT_API} Error pinging server: {e}", flush=True)
        #         return {"success": False, "message": ""}

        # Tell client we are ready to accept requests
        @app.get("/v1/connect")
        def connect() -> classes.ConnectResponse:
            return {
                "success": True,
                "message": f"Connected to api server on port {self.SERVER_PORT}. Refer to '{self.XHR_PROTOCOL}://localhost:{self.SERVER_PORT}/docs' for api docs.",
                "data": {
                    "docs": f"{self.XHR_PROTOCOL}://localhost:{self.SERVER_PORT}/docs",
                    "version": self.api_version,
                },
            }

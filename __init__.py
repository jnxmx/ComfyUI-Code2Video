import logging
from .nodes import (
    SenseNovaCode2VideoLoader,
    SenseNovaCode2VideoSettings,
    SenseNovaCode2VideoStudio,
)
from .core.server_routes import register_server_routes

logger = logging.getLogger("ComfyUI-Code2Video")

NODE_CLASS_MAPPINGS = {
    "SenseNovaCode2VideoLoader": SenseNovaCode2VideoLoader,
    "SenseNovaCode2VideoSettings": SenseNovaCode2VideoSettings,
    "SenseNovaCode2VideoStudio": SenseNovaCode2VideoStudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SenseNovaCode2VideoLoader": "SenseNova Code2Video Loader (INT8)",
    "SenseNovaCode2VideoSettings": "SenseNova Code2Video Settings",
    "SenseNovaCode2VideoStudio": "SenseNova Code2Video Studio",
}

WEB_DIRECTORY = "./web"

# Initialize custom HTTP and WebSocket API routes
register_server_routes()

logger.info("ComfyUI-Code2Video custom nodes initialized successfully.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

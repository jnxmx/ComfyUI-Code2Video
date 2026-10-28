import os
import mimetypes
import logging
from aiohttp import web
import folder_paths
from core.stateful_runner import _SessionStore

logger = logging.getLogger("ComfyUI-Code2Video")

WORKSPACE_ROOT = os.path.join(folder_paths.get_temp_directory(), "code2video")


def register_server_routes():
    """Registers API and static preview routes onto PromptServer.instance.app."""
    try:
        from server import PromptServer
        if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
            logger.warning("PromptServer instance not ready; deferring route registration.")
            return

        routes = PromptServer.instance.routes

        # 1. Preview static asset route
        @routes.get(r"/code2video/preview/{session_id}/{filepath:.*}")
        async def handle_preview_asset(request):
            session_id = request.match_info.get("session_id", "").strip()
            filepath = request.match_info.get("filepath", "index.html").strip()

            if not session_id:
                return web.Response(status=400, text="Missing session_id")

            base_dir = os.path.normpath(os.path.join(WORKSPACE_ROOT, session_id))
            target_path = os.path.normpath(os.path.join(base_dir, filepath))

            # Security check: prevent directory traversal
            if not target_path.startswith(base_dir):
                return web.Response(status=403, text="Forbidden path traversal")

            if not os.path.isfile(target_path):
                return web.Response(status=404, text=f"File not found: {filepath}")

            mime_type, _ = mimetypes.guess_type(target_path)
            if not mime_type:
                mime_type = "application/octet-stream"
            if filepath.endswith(".html"):
                mime_type = "text/html; charset=utf-8"
            elif filepath.endswith(".js") or filepath.endswith(".mjs"):
                mime_type = "text/javascript; charset=utf-8"

            headers = {
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
            }

            try:
                with open(target_path, "rb") as f:
                    content = f.read()
                return web.Response(body=content, content_type=mime_type, headers=headers)
            except Exception as e:
                logger.error(f"Error serving preview file {target_path}: {e}")
                return web.Response(status=500, text=str(e))

        # 2. Control endpoint (Pause / Resume / Refine)
        @routes.post("/code2video/control")
        async def handle_control(request):
            try:
                data = await request.json()
            except Exception:
                return web.json_response({"success": False, "error": "Invalid JSON"}, status=400)

            session_id = data.get("session_id", "").strip()
            action = data.get("action", "").lower().strip()

            session = _SessionStore.get(session_id)
            if not session:
                return web.json_response({"success": False, "error": f"Session {session_id} not found"}, status=404)

            if action == "pause":
                session.request_pause()
                logger.info(f"Session {session_id} paused by user request.")
                return web.json_response({"success": True, "status": "paused", "session_id": session_id})
            elif action == "resume":
                session.reset_pause()
                logger.info(f"Session {session_id} resumed.")
                return web.json_response({"success": True, "status": "resumed", "session_id": session_id})
            else:
                return web.json_response({"success": False, "error": f"Unknown action: {action}"}, status=400)

        # 3. Session info and file tree endpoint
        @routes.get("/code2video/session/{session_id}")
        async def handle_session_info(request):
            session_id = request.match_info.get("session_id", "").strip()
            session = _SessionStore.get(session_id)
            if not session:
                return web.json_response({"success": False, "error": "Session not found"}, status=404)

            # Collect files list
            file_tree = []
            if os.path.isdir(session.harness.workspace_dir):
                for root, _, files in os.walk(session.harness.workspace_dir):
                    for file in files:
                        full_p = os.path.join(root, file)
                        rel_p = os.path.relpath(full_p, session.harness.workspace_dir)
                        file_tree.append(rel_p)

            return web.json_response({
                "success": True,
                "session_id": session.session_id,
                "status": session.status,
                "turn_index": session.turn_index,
                "preview_ready": session.preview_ready,
                "preview_url": f"/code2video/preview/{session.session_id}/index.html" if session.preview_ready else None,
                "files": file_tree,
                "last_code": session.last_code,
            })

        logger.info("ComfyUI-Code2Video preview and control routes registered.")
    except Exception as e:
        logger.warning(f"Could not register server routes: {e}")

import os
import re
import shutil
import subprocess
import logging
from typing import Dict, Any, Tuple, Optional
import torch
import folder_paths

logger = logging.getLogger("ComfyUI-Code2Video")

WORKSPACE_ROOT = os.path.join(folder_paths.get_temp_directory(), "code2video")


class ToolHarness:
    """Manages the sandboxed project workspace and tool execution for a session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.workspace_dir = os.path.join(WORKSPACE_ROOT, session_id)
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.assets_dir = os.path.join(self.workspace_dir, "assets")
        os.makedirs(self.assets_dir, exist_ok=True)

    def _safe_path(self, rel_path: str) -> str:
        """Ensures the target path does not escape the workspace directory."""
        clean_rel = os.path.normpath(rel_path.strip().lstrip("/\\"))
        full_path = os.path.normpath(os.path.join(self.workspace_dir, clean_rel))
        if not full_path.startswith(self.workspace_dir):
            raise ValueError(f"Path escapes workspace: {rel_path}")
        return full_path

    def stage_asset(self, filename: str, data: bytes) -> str:
        """Saves a binary asset into the workspace assets directory."""
        dest_path = os.path.join(self.assets_dir, filename)
        with open(dest_path, "wb") as f:
            f.write(data)
        return os.path.relpath(dest_path, self.workspace_dir)

    def write_file(self, rel_path: str, content: str) -> str:
        """Executes the Write tool."""
        full_path = self._safe_path(rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {rel_path}"

    def edit_file(self, rel_path: str, target: str, replacement: str) -> str:
        """Executes the Edit tool."""
        full_path = self._safe_path(rel_path)
        if not os.path.isfile(full_path):
            return f"Error: File not found: {rel_path}"
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        if target not in content:
            return f"Error: Target string not found in {rel_path}."

        new_content = content.replace(target, replacement)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Successfully replaced target in {rel_path}."

    def read_file(self, rel_path: str) -> str:
        """Executes the Read tool."""
        full_path = self._safe_path(rel_path)
        if not os.path.isfile(full_path):
            return f"Error: File not found: {rel_path}"
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    def run_bash(self, command: str, timeout: int = 60) -> str:
        """Executes a bash command inside the workspace directory."""
        try:
            res = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = res.stdout
            err = res.stderr
            code = res.returncode
            result = f"Exit code: {code}\n"
            if out:
                result += f"Stdout:\n{out}\n"
            if err:
                result += f"Stderr:\n{err}\n"
            return result
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds."
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def execute_tool(self, func_name: str, params: Dict[str, str]) -> str:
        """Dispatches tool call to appropriate method."""
        func = func_name.strip()
        if func == "Write":
            path = params.get("path", "").strip()
            content = params.get("content", "")
            return self.write_file(path, content)
        elif func == "Edit":
            path = params.get("path", "").strip()
            target = params.get("target", "")
            replacement = params.get("replacement", "")
            return self.edit_file(path, target, replacement)
        elif func == "Read":
            path = params.get("path", "").strip()
            return self.read_file(path)
        elif func == "Bash":
            cmd = params.get("command", "").strip()
            return self.run_bash(cmd)
        else:
            return f"Error: Unknown tool '{func}'."

    def render_to_video(
        self,
        fps: int = 30,
        duration: float = 5.0,
        width: int = 1280,
        height: int = 720,
    ) -> Optional[str]:
        """
        Attempts to render the workspace index.html to MP4 using capture.mjs or available headless renderer.
        Returns the path to the output video file if successful.
        """
        index_html = os.path.join(self.workspace_dir, "index.html")
        if not os.path.isfile(index_html):
            return None

        out_video = os.path.join(self.workspace_dir, "out", "video.mp4")
        if os.path.isfile(out_video):
            return out_video

        # Check if capture script is available
        plugin_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        capture_script = os.path.join(plugin_root, "skills", "code-video", "scripts", "capture.mjs")

        if os.path.isfile(capture_script) and shutil.which("node"):
            cmd = (
                f"node \"{capture_script}\" \"{self.workspace_dir}\" "
                f"--page index.html --fps {fps} --dur {duration} --size {width}x{height} "
                f"--out out/video.mp4"
            )
            logger.info(f"Rendering video with capture.mjs: {cmd}")
            res = self.run_bash(cmd, timeout=120)
            logger.info(res)
            if os.path.isfile(out_video):
                return out_video

        return None

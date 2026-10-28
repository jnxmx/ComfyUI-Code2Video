import os
import re
import uuid
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import torch

import comfy.model_management
from core.tool_harness import ToolHarness

logger = logging.getLogger("ComfyUI-Code2Video")


def find_transformer_with_sampler(obj: Any) -> Optional[Any]:
    """Locates the underlying transformer module that implements sample_token."""
    if hasattr(obj, "sample_token"):
        return obj
    if hasattr(obj, "transformer") and hasattr(obj.transformer, "sample_token"):
        return obj.transformer
    csm = getattr(obj, "cond_stage_model", None)
    if csm:
        if hasattr(csm, "transformer") and hasattr(csm.transformer, "sample_token"):
            return csm.transformer
        for _, ch in csm.named_children():
            if hasattr(ch, "transformer") and hasattr(ch.transformer, "sample_token"):
                return ch.transformer
            if hasattr(ch, "sample_token"):
                return ch
    m = getattr(obj, "model", None)
    if m:
        return find_transformer_with_sampler(m)
    return None


def send_server_event(event_name: str, data: Dict[str, Any]):
    """Sends a WebSocket message to the ComfyUI frontend if PromptServer is active."""
    try:
        from server import PromptServer
        if hasattr(PromptServer, "instance") and PromptServer.instance is not None:
            PromptServer.instance.send_sync(event_name, data)
    except Exception as e:
        logger.debug(f"Could not send event {event_name}: {e}")


class Code2VideoSession:
    """Maintains state, chat history, and workspace context for a Code2Video studio session."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())[:12]
        self.harness = ToolHarness(self.session_id)
        self.messages: List[Dict[str, str]] = []
        self.turn_index = 0
        self.status = "idle"  # idle, running, paused, completed, error
        self.pause_requested = False
        self.preview_ready = False
        self.last_code = ""

    def reset_pause(self):
        self.pause_requested = False

    def request_pause(self):
        self.pause_requested = True
        self.status = "paused"


class _SessionStore:
    """In-memory singleton registry of active sessions."""
    _sessions: Dict[str, Code2VideoSession] = {}

    @classmethod
    def get_or_create(cls, session_id: Optional[str] = None) -> Code2VideoSession:
        if not session_id or session_id.strip() == "":
            sess = Code2VideoSession()
            cls._sessions[sess.session_id] = sess
            return sess
        if session_id not in cls._sessions:
            cls._sessions[session_id] = Code2VideoSession(session_id)
        return cls._sessions[session_id]

    @classmethod
    def get(cls, session_id: str) -> Optional[Code2VideoSession]:
        return cls._sessions.get(session_id)

    @classmethod
    def delete(cls, session_id: str):
        if session_id in cls._sessions:
            del cls._sessions[session_id]


def parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, str]]]:
    """
    Extracts tool calls from generated text. Supports SenseNova / Qwen syntax:
    <tool_call>
    <function=Name>
    <parameter=key>value</parameter>
    </tool_call>
    """
    tool_calls = []

    # 1. Standard SenseNova / Qwen tool call block
    pattern = re.compile(r"<tool_call>\s*<function=([A-Za-z0-9_-]+)>(.*?)</tool_call>", re.DOTALL)
    param_pattern = re.compile(r"<parameter=([A-Za-z0-9_-]+)>(.*?)</parameter>", re.DOTALL)

    matches = pattern.findall(text)
    for func_name, body in matches:
        params = {}
        for p_name, p_val in param_pattern.findall(body):
            params[p_name] = p_val.strip()
        tool_calls.append((func_name.strip(), params))

    # 2. Fallback: markdown code blocks if model wrote raw index.html code directly
    if not tool_calls and "```html" in text:
        html_blocks = re.findall(r"```html\s*\n(.*?)\n```", text, re.DOTALL)
        if html_blocks:
            tool_calls.append(("Write", {"path": "index.html", "content": html_blocks[0].strip()}))

    return tool_calls


def format_chatml_prompt(messages: List[Dict[str, str]]) -> str:
    """Formats messages into Qwen/ChatML prompt string."""
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted


def run_session_turn(
    session: Code2VideoSession,
    pipeline: Any,
    system_prompt: str,
    user_instruction: str,
    vision_tensors: Optional[List[torch.Tensor]] = None,
    max_turns: int = 5,
    max_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:
    """
    Executes an interactive agentic turn loop with Qwen3.5/SenseNova,
    handling streaming, tool calls, preview updates, and pause requests.
    """
    session.status = "running"
    session.reset_pause()
    send_server_event("code2video_status", {
        "session_id": session.session_id,
        "status": "running",
        "turn": session.turn_index,
    })

    # Initialize system message if first turn
    if not session.messages:
        session.messages.append({"role": "system", "content": system_prompt})

    # Append user instruction
    session.messages.append({"role": "user", "content": user_instruction})

    current_turn = 0
    final_assistant_response = ""

    while current_turn < max_turns:
        # Check for interrupt / cancel from ComfyUI
        comfy.model_management.throw_exception_if_processing_interrupted()

        # Check for pause requested via server route
        if session.pause_requested:
            session.status = "paused"
            send_server_event("code2video_status", {
                "session_id": session.session_id,
                "status": "paused",
                "turn": session.turn_index,
            })
            break

        # Format input prompt
        chatml_text = format_chatml_prompt(session.messages)

        # Tokenize (supports comfy.sd.CLIP or SenseNovaPipeline)
        if hasattr(pipeline, "tokenize"):
            tokens = pipeline.tokenize(chatml_text)
        elif hasattr(pipeline, "tokenizer"):
            tokens = pipeline.tokenizer.tokenize_with_weights(chatml_text)
        else:
            raise TypeError(f"Provided model/clip object has no tokenizer: {type(pipeline)}")

        # Generate tokens with live Thinking & Burning Preview hook
        logger.info(f"Generating tokens for session {session.session_id} (turn {current_turn + 1})...")
        send_server_event("code2video_status", {
            "session_id": session.session_id,
            "status": "thinking",
            "turn": current_turn + 1,
        })

        generated_text = ""
        decoder_fn = getattr(pipeline, "decode", None)
        if decoder_fn is None and hasattr(pipeline, "tokenizer"):
            decoder_fn = getattr(pipeline.tokenizer, "decode", None)

        start_time = time.time()
        burn_count = [0]
        in_thought = [False]
        transformer = find_transformer_with_sampler(pipeline)
        orig_sample_token = None

        if transformer is not None and hasattr(transformer, "sample_token"):
            orig_sample_token = transformer.sample_token

            def hooked_sample_token(*args, **kwargs):
                next_tok = orig_sample_token(*args, **kwargs)
                try:
                    tok_id = next_tok.item() if next_tok.numel() == 1 else next_tok[0, 0].item()
                    burn_count[0] += 1
                    chunk = decoder_fn([tok_id]) if decoder_fn else ""
                    elapsed = max(0.001, time.time() - start_time)
                    tps = burn_count[0] / elapsed

                    lower = chunk.lower()
                    if "<think>" in lower or "<thought>" in lower:
                        in_thought[0] = True
                    elif "</think>" in lower or "</thought>" in lower:
                        in_thought[0] = False

                    send_server_event("code2video_thinking", {
                        "session_id": session.session_id,
                        "token_id": tok_id,
                        "chunk": chunk,
                        "burn_count": burn_count[0],
                        "tps": round(tps, 1),
                        "is_thinking": in_thought[0],
                        "elapsed": round(elapsed, 1),
                    })
                except Exception as err:
                    logger.debug(f"Error in thinking stream hook: {err}")
                return next_tok

            transformer.sample_token = hooked_sample_token

        try:
            if hasattr(pipeline, "generate"):
                gen_output = pipeline.generate(
                    tokens,
                    do_sample=(temperature > 0.0),
                    max_length=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=0.0,
                    repetition_penalty=1.05,
                    seed=42,
                )
            elif hasattr(pipeline, "model") and hasattr(pipeline.model, "generate"):
                gen_output = pipeline.model.generate(
                    tokens,
                    do_sample=(temperature > 0.0),
                    max_length=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=0.0,
                    repetition_penalty=1.05,
                    seed=42,
                )
            else:
                raise TypeError(f"Provided model/clip object has no generate method: {type(pipeline)}")

            if isinstance(gen_output, list) and len(gen_output) > 0:
                gen_ids = gen_output[0]
                generated_text = decoder_fn(gen_ids) if decoder_fn else str(gen_ids)
            elif isinstance(gen_output, torch.Tensor):
                token_list = gen_output.squeeze().tolist()
                generated_text = decoder_fn(token_list) if decoder_fn else str(token_list)
            elif isinstance(gen_output, str):
                generated_text = gen_output
            elif isinstance(gen_output, tuple) and len(gen_output) > 0:
                first_item = gen_output[0]
                if isinstance(first_item, (list, torch.Tensor)):
                    ids = first_item if isinstance(first_item, list) else first_item.squeeze().tolist()
                    generated_text = decoder_fn(ids) if decoder_fn else str(ids)
                elif isinstance(first_item, str):
                    generated_text = first_item

        except Exception as e:
            logger.error(f"Error during token generation: {e}")
            session.status = "error"
            send_server_event("code2video_status", {
                "session_id": session.session_id,
                "status": "error",
                "error": str(e),
            })
            raise e
        finally:
            if transformer is not None and orig_sample_token is not None:
                transformer.sample_token = orig_sample_token

        # Stream generated response to UI
        send_server_event("code2video_token", {
            "session_id": session.session_id,
            "token": generated_text,
            "role": "assistant",
        })

        session.messages.append({"role": "assistant", "content": generated_text})
        final_assistant_response = generated_text

        # Parse tool calls
        tool_calls = parse_tool_calls(generated_text)
        if not tool_calls:
            # Model finished turn without tool calls
            logger.info("No tool calls detected; turn completed.")
            break

        # Execute tool calls sequentially
        tool_outputs = []
        for func_name, params in tool_calls:
            send_server_event("code2video_tool_executing", {
                "session_id": session.session_id,
                "function": func_name,
                "params": params,
            })

            result = session.harness.execute_tool(func_name, params)
            tool_outputs.append(f"Tool {func_name} output:\n{result}")

            # If index.html was created or updated, notify frontend of live preview
            index_path = os.path.join(session.harness.workspace_dir, "index.html")
            if os.path.isfile(index_path):
                session.preview_ready = True
                try:
                    with open(index_path, "r", encoding="utf-8") as f:
                        session.last_code = f.read()
                except Exception:
                    pass

                preview_url = f"/code2video/preview/{session.session_id}/index.html"
                send_server_event("code2video_preview_ready", {
                    "session_id": session.session_id,
                    "url": preview_url,
                    "code": session.last_code,
                })

        # Append tool outputs to chat history for next turn
        combined_tool_output = "\n\n".join(tool_outputs)
        session.messages.append({"role": "tool", "content": combined_tool_output})

        current_turn += 1

    session.turn_index += 1
    if not session.pause_requested:
        session.status = "completed"
        send_server_event("code2video_status", {
            "session_id": session.session_id,
            "status": "completed",
            "turn": session.turn_index,
        })

    return final_assistant_response

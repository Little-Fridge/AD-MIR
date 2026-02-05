"""
Admir ReAct Agent - Iterative Think-Act-Observe Loop
"""

import json
import os
import re
import traceback
import httpx
import logging
import sys
import datetime
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type, before_sleep_log
from typing import List, Dict, Tuple, Annotated as A
from openai import OpenAI

from admir.build_database import (
    frame_inspect_tool as frame_inspect_tool_impl,
    clip_search_tool as clip_search_tool_impl,
    global_browse_tool as global_browse_tool_impl,
    init_single_video_db
)
from admir.func_call_shema import as_json_schema, doc as D
from admir import config

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger(__name__)

# =====================
# Configuration
# =====================
BASE_URL = config.LOCAL_VLLM_BASE_URL
API_KEY = config.OPENAI_API_KEY

AGENT_MODEL_NAME = "gpt-4o"
EXPERT_MODEL_NAME = "o1"
REFINE_MODEL_NAME = "gpt-4o-mini"

# Client initialization
gpt_client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    max_retries=1,
    http_client=httpx.Client(base_url=BASE_URL, follow_redirects=True, timeout=120.0),
)

local_expert_client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    http_client=httpx.Client(timeout=120.0)
)

import openai
@retry(
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.APIStatusError)),
    wait=wait_random_exponential(multiplier=2, max=60),
    stop=stop_after_attempt(50), 
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def call_openai_with_retry(**kwargs):
    """Wrapper for OpenAI API calls with retry logic."""
    try:
        return gpt_client.chat.completions.create(**kwargs)
    except openai.APIStatusError as e:
        if e.status_code == 429:
            print(f"⚠️ [429 Limit] Waiting... (Error: {e})")
        raise e

class TaskCompleted(Exception):
    def __init__(self, answer: str):
        self.answer = answer
        super().__init__(answer)

def finish_with_answer(
    answer: A[str, D("The final concise answer.")],
    evidence_check: A[str, D("Explicitly state the EVIDENCE that supports your answer.")]
) -> None:
    raise TaskCompleted(answer)

# =====================================================================
# Expert Tools
# =====================================================================

def _call_expert_model(system_prompt: str, user_content: str | list, max_tokens: int = None) -> str:
    """Call the expert model (e.g., Qwen/Gemini/O1) with multimodal input."""
    target_model = EXPERT_MODEL_NAME 
    current_max_tokens = max_tokens if max_tokens else 10000

    messages = [{"role": "system", "content": system_prompt}]
    if isinstance(user_content, str):
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_content})
    
    print(f"  🧠 [Expert] Calling {target_model}...")
    try:
        response = local_expert_client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=0,
            max_tokens=current_max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty content returned")
        return content
    except Exception as e:
        print(f"  ⚠️ [Expert Fail] {target_model} failed: {str(e)}")
        return f"[Error calling expert model: {str(e)}]"

def communication_expert_tool_impl(
    database,
    query_focus: str,
    start_time: str = "00:00:00",
    end_time: str = "end",
    global_context: str = "" 
) -> str:
    """Performs grid-based visual analysis of frames."""
    import numpy as np
    import glob
    import base64
    import io
    from PIL import Image
    from admir.build_database import _get_multimodal_context

    def stitch_images_grid(image_paths, grid_size=(2, 2)):
        images = []
        for p in image_paths:
            try:
                img = Image.open(p)
                images.append(img)
            except Exception as e:
                print(f"[Warn] Failed to load image {p}: {e}")
        
        if not images: return None
        w, h = images[0].size
        grid_w, grid_h = grid_size
        new_im = Image.new('RGB', (w * grid_w, h * grid_h), (0, 0, 0))
        
        for idx, im in enumerate(images):
            if im.size != (w, h): im = im.resize((w, h))
            row = idx // grid_w
            col = idx % grid_w
            new_im.paste(im, (col * w, row * h))
            
        buffered = io.BytesIO()
        new_im.save(buffered, format="JPEG", quality=85) 
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    video_meta = database.get_additional_data()
    video_root = video_meta.get("video_file_root", "")
    frames_dir = os.path.join(video_root, "frames")
    all_frames = sorted(glob.glob(os.path.join(frames_dir, "frame_n*.jpg")))
    
    target_total_frames = 64
    if len(all_frames) > target_total_frames:
        indices = np.linspace(0, len(all_frames) - 1, target_total_frames, dtype=int)
        selected_frames = [all_frames[i] for i in indices]
    else:
        selected_frames = all_frames

    mm_context = _get_multimodal_context(database)
    raw_asr = mm_context.get("asr_formatted", "No audio.")

    system_prompt = "You are an Elite Advertising Forensics Expert. Analyze the 2x2 Grid Images."
    
    user_content_list = []
    user_content_list.append({"type": "text", "text": f"Question: {query_focus}\nContext: {global_context}\nASR: {raw_asr}"})

    batch_size = 4
    batched_frames = [selected_frames[i:i + batch_size] for i in range(0, len(selected_frames), batch_size)]
    
    for batch in batched_frames:
        if not batch: continue
        b64_grid = stitch_images_grid(batch)
        if b64_grid:
            user_content_list.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{b64_grid}", "detail": "high"}
            })

    return _call_expert_model(system_prompt, user_content_list, max_tokens=2048)

# =====================================================================
# Agent Implementation
# =====================================================================

class AdmirAgent:
    def __init__(
        self, 
        video_db_path: str, 
        video_caption_path: str,
        max_iterations: int = 8,
        embedding_dim: int = 1024,
    ):
        if "ADMIR_EMBEDDING_URL" not in os.environ:
            os.environ["ADMIR_EMBEDDING_URL"] = "http://0.0.0.0:8090"
            
        self.video_db = init_single_video_db(video_caption_path, video_db_path, embedding_dim)
        self.max_iterations = max_iterations
        self._build_tool_registry()
        
    def _build_tool_registry(self):
        def global_browse_tool(query: A[str, D("Question to browse")]) -> str:
            return global_browse_tool_impl(self.video_db, query)
        
        def clip_search_tool(query: A[str, D("Event to search")], top_k: int = 5) -> str:
            return clip_search_tool_impl(self.video_db, query, top_k)
        
        def frame_inspect_tool(question: str, time_ranges_hhmmss: List[List[str]] = None, analysis_mode: str = "literal") -> str:
            if not time_ranges_hhmmss: time_ranges_hhmmss = None 
            try:
                time_tuples = [tuple(tr) for tr in time_ranges_hhmmss] if time_ranges_hhmmss else None
                return frame_inspect_tool_impl(self.video_db, question, time_tuples, analysis_mode)
            except Exception as e: return str(e)

        def communication_expert_tool(query_focus: str, start_time: str = "00:00:00", end_time: str = "end") -> str:
            ctx = getattr(self, 'current_global_context', "")
            return communication_expert_tool_impl(self.video_db, query_focus, start_time, end_time, ctx)
                
        self.tools = [global_browse_tool, clip_search_tool, frame_inspect_tool, communication_expert_tool, finish_with_answer]
        self.tool_map = {tool.__name__: tool for tool in self.tools}
        self.tool_schemas = [{"type": "function", "function": as_json_schema(tool)} for tool in self.tools]

    def _get_react_system_prompt(self) -> str:
        return """You are an advanced Video Analysis Agent.
        1. TRUST communication_expert_tool as PRIMARY TRUTH.
        2. Do NOT verify Expert findings with frame_inspect unless Expert fails.
        3. If Expert answers fully, finish immediately.
        Output: THOUGHT then ACTION."""

    def run(self, question: str) -> Dict:
        print(f"🤖 Admir Agent: {question}")
        
        # Pre-emptive browse
        try:
            global_ctx = global_browse_tool_impl(self.video_db, "Summary of narrative")
            self.current_global_context = json.loads(global_ctx).get('overview', '')
        except: 
            self.current_global_context = ""

        messages = [
            {"role": "system", "content": self._get_react_system_prompt() + f"\nGlobal Context: {self.current_global_context}"},
            {"role": "user", "content": f"Question: {question}"},
        ]
        
        iteration_history = []
        
        try:
            for iteration in range(self.max_iterations):
                print(f"ITERATION {iteration + 1}")
                response = call_openai_with_retry(
                    model=AGENT_MODEL_NAME, messages=messages, tools=self.tool_schemas, tool_choice="auto", temperature=0
                )
                assistant_msg = response.choices[0].message
                
                # Manual dictionary conversion to prevent serialization errors
                msg_dict = {"role": assistant_msg.role, "content": assistant_msg.content}
                if assistant_msg.tool_calls:
                    msg_dict["tool_calls"] = [{
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    } for tc in assistant_msg.tool_calls]
                
                messages.append(msg_dict)

                if not assistant_msg.tool_calls:
                    messages.append({"role": "user", "content": "Please use a tool or finish."})
                    continue

                for tool_call in assistant_msg.tool_calls:
                    name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except:
                        args = {}
                    
                    if name == "finish_with_answer":
                        raise TaskCompleted(args.get('answer', ''))
                    
                    if name in self.tool_map:
                        result = self.tool_map[name](**args)
                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)[:4000]})
                        iteration_history.append({"iter": iteration, "tool": name, "result": str(result)[:100]})

        except TaskCompleted as e:
            return {"answer": e.answer, "history": iteration_history}
        except Exception as e:
            return {"answer": "Error", "error": str(e)}
            
        return {"answer": "Max iterations", "history": iteration_history}
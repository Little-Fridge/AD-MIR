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
try:
    from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type, before_sleep_log
except ImportError:
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def stop_after_attempt(*args, **kwargs):
        return None
    def wait_random_exponential(*args, **kwargs):
        return None
    def retry_if_exception_type(*args, **kwargs):
        return None
    def before_sleep_log(*args, **kwargs):
        return None
from typing import List, Dict, Tuple, Annotated as A
from openai import OpenAI

from admir.build_database import (
    frame_inspect_tool as frame_inspect_tool_impl,
    clip_search_tool as clip_search_tool_impl,
    global_browse_tool as global_browse_tool_impl,
    get_active_subject_registry,
    init_single_video_db
)
from admir.func_call_schema import as_json_schema, doc as D
from admir.utils import create_chat_completion, robust_json_parse
from admir import config

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger(__name__)

# =====================
# Configuration
# =====================
BASE_URL = config.LOCAL_VLLM_BASE_URL
API_KEY = config.OPENAI_API_KEY
if getattr(config, "STRICT_PAPER_MODE", False) and not API_KEY:
    raise RuntimeError(
        "ADMIR_STRICT_PAPER_MODE=1 requires OPENAI_API_KEY, or an explicitly "
        "configured local endpoint that accepts the placeholder key."
    )
CLIENT_API_KEY = API_KEY or "EMPTY"

AGENT_MODEL_NAME = config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME
EXPERT_MODEL_NAME = getattr(config, "AOAI_COMMUNICATION_EXPERT_MODEL_NAME", config.AOAI_TOOL_VLM_MODEL_NAME)
REFINE_MODEL_NAME = config.AOAI_REFINE_LLM_MODEL_NAME

# Client initialization
REQUEST_TIMEOUT = float(os.environ.get("ADMIR_OPENAI_TIMEOUT", "120.0"))

gpt_client = OpenAI(
    base_url=BASE_URL,
    api_key=CLIENT_API_KEY,
    max_retries=1,
    http_client=httpx.Client(base_url=BASE_URL, follow_redirects=True, timeout=REQUEST_TIMEOUT),
)

local_expert_client = OpenAI(
    base_url=BASE_URL,
    api_key=CLIENT_API_KEY,
    http_client=httpx.Client(timeout=REQUEST_TIMEOUT)
)

import openai
@retry(
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.APIStatusError)),
    wait=wait_random_exponential(multiplier=2, max=60),
    stop=stop_after_attempt(8), 
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
    def __init__(self, answer: str, evidence_check: str = ""):
        self.answer = answer
        self.evidence_check = evidence_check
        super().__init__(answer)

def finish_with_answer(
    answer: A[str, D("The final concise answer.")],
    evidence_check: A[str, D("Explicitly state the EVIDENCE that supports your answer.")]
) -> None:
    raise TaskCompleted(answer, evidence_check)

# =====================================================================
# Expert Tools
# =====================================================================

def _call_expert_model(system_prompt: str, user_content: str | list, max_tokens: int = None) -> str:
    """Call the paper communication expert model with multimodal input."""
    target_model = EXPERT_MODEL_NAME 
    default_max_tokens = max_tokens if max_tokens else 10000
    current_max_tokens = int(os.environ.get("ADMIR_EXPERT_MAX_TOKENS", default_max_tokens))

    messages = [{"role": "system", "content": system_prompt}]
    if isinstance(user_content, str):
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_content})
    
    print(f"  🧠 [Expert] Calling {target_model}...")
    try:
        response = create_chat_completion(
            local_expert_client,
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
        raise RuntimeError(f"Expert model call failed: {e}") from e

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
                img = Image.open(p).convert("RGB")
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

        # Keep 64-frame coverage while bounding visual tokens for local VLM
        # backends with finite context windows.
        max_side = int(os.environ.get("ADMIR_EXPERT_GRID_MAX_SIDE", "1024"))
        if max_side > 0:
            longest_side = max(new_im.size)
            if longest_side > max_side:
                scale = max_side / float(longest_side)
                new_size = (
                    max(1, int(round(new_im.size[0] * scale))),
                    max(1, int(round(new_im.size[1] * scale))),
                )
                resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                new_im = new_im.resize(new_size, resample)
            
        buffered = io.BytesIO()
        jpeg_quality = int(os.environ.get("ADMIR_EXPERT_GRID_JPEG_QUALITY", "85"))
        new_im.save(buffered, format="JPEG", quality=jpeg_quality)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _ts_to_seconds(ts: str, fallback_end: float) -> float:
        if str(ts).lower() == "end":
            return fallback_end
        parts = str(ts).split(":")
        try:
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(ts)
        except Exception:
            return 0.0

    video_meta = database.get_additional_data()
    video_root = video_meta.get("video_file_root", "")
    fps = float(video_meta.get("fps", getattr(config, "VIDEO_FPS", 1)) or 1)
    frames_dir = os.path.join(video_root, "frames")
    all_frames = sorted(glob.glob(os.path.join(frames_dir, "frame_n*.jpg")))
    if not all_frames:
        return "Error: No frame files found for communication expert."

    video_end = (len(all_frames) - 1) / max(fps, 1e-6)
    start_s = max(0.0, _ts_to_seconds(start_time, video_end))
    end_s = min(video_end, _ts_to_seconds(end_time, video_end))
    if start_s > end_s:
        start_s, end_s = end_s, start_s

    start_idx = max(0, min(len(all_frames) - 1, int(round(start_s * fps))))
    end_idx = max(0, min(len(all_frames) - 1, int(round(end_s * fps))))
    candidate_frames = all_frames[start_idx:end_idx + 1]

    target_total_frames = int(getattr(config, "EXPERT_MAX_GRID_FRAMES", 64))
    if len(candidate_frames) > target_total_frames:
        indices = np.linspace(0, len(candidate_frames) - 1, target_total_frames, dtype=int)
        selected_frames = [candidate_frames[i] for i in indices]
    else:
        selected_frames = candidate_frames

    mm_context = _get_multimodal_context(database)
    raw_asr = mm_context.get("asr_formatted", "No audio.")
    raw_ocr = mm_context.get("ocr_formatted", "")
    active_subjects = get_active_subject_registry(database, query_focus, global_context, top_k=3)

    system_prompt = """You are an Elite Advertising Forensics Expert & Visual Semiotics Analyst.
**YOUR MISSION**: Decode the provided advertising video segment to uncover narrative structure, character relationships, and persuasive strategy.
**INPUT FORMAT (CRITICAL)**:
- The visual input consists of 2x2 Grid Images.
- Each image contains 4 chronological video frames.
- Reading Order: Top-Left -> Top-Right -> Bottom-Left -> Bottom-Right.
**CORE ANALYSIS PROTOCOLS**:
1. OCR & BRAND TRUTH (HIGHEST PRIORITY): Any text on screen is fact. Identify logos.
2. UNIVERSAL CHARACTER DYNAMICS: Analyze transactional, conflict, or affectionate interactions.
3. NARRATIVE ARC: Hook -> Problem -> Product -> CTA.
4. GROUNDING: Do not invent objects not present in the grids."""
    
    user_content_list = []
    user_content_list.append({
        "type": "text",
        "text": (
            f"Question: {query_focus}\n"
            f"Segment: {start_s:.1f}s to {end_s:.1f}s\n"
            f"Global Context: {global_context}\n"
            f"ASR: {raw_asr}\n"
            f"OCR/Text: {raw_ocr}\n"
            f"Active Subject Registry: {json.dumps(active_subjects, ensure_ascii=False)}"
        )
    })

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
        max_iterations: int = None,
        embedding_dim: int = 1024,
    ):
        validate = getattr(config, "validate_strict_paper_config", None)
        if callable(validate):
            validate(require_api_key=True)
        if getattr(config, "EMBEDDING_ENDPOINT", "") and "ADMIR_EMBEDDING_URL" not in os.environ:
            os.environ["ADMIR_EMBEDDING_URL"] = config.EMBEDDING_ENDPOINT
            
        self.video_db = init_single_video_db(video_caption_path, video_db_path, embedding_dim)
        self.max_iterations = max_iterations or config.MAX_ITERATIONS
        self.time_range_history = []
        self._redirect_toggle = False
        self._literal_verification_cache = {}
        self._build_tool_registry()

    @staticmethod
    def _hhmmss_to_seconds(ts: str, fallback_end: float = 0.0) -> float:
        if str(ts).lower() == "end":
            return fallback_end
        parts = str(ts).split(":")
        try:
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(ts)
        except Exception:
            return 0.0

    @staticmethod
    def _seconds_to_hhmmss(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

    def _video_length_seconds(self) -> float:
        meta = self.video_db.get_additional_data()
        return self._hhmmss_to_seconds(meta.get("video_length", "0"), 0.0)

    def _normalize_ranges(self, ranges: List[List[str]] | None) -> List[Tuple[float, float]]:
        if not ranges:
            return []
        video_end = self._video_length_seconds()
        normalized = []
        for tr in ranges:
            if not tr or len(tr) < 2:
                continue
            start = self._hhmmss_to_seconds(tr[0], video_end)
            end = self._hhmmss_to_seconds(tr[1], video_end)
            if start > end:
                start, end = end, start
            normalized.append((start, end))
        return normalized

    def _maybe_redirect_stagnant_ranges(self, ranges: List[List[str]] | None) -> List[List[str]] | None:
        normalized = self._normalize_ranges(ranges)
        if not normalized:
            return ranges

        overlap_threshold = float(getattr(config, "TEMPORAL_STAGNATION_OVERLAP", 0.6))
        repeat_threshold = int(getattr(config, "TEMPORAL_STAGNATION_REPEATS", 2))

        stagnant_hits = 0
        for current_start, current_end in normalized:
            duration = max(current_end - current_start, 1e-6)
            for hist_start, hist_end in self.time_range_history:
                intersection = max(0.0, min(current_end, hist_end) - max(current_start, hist_start))
                if intersection / duration > overlap_threshold:
                    stagnant_hits += 1

        self.time_range_history.extend(normalized)
        if stagnant_hits < repeat_threshold:
            return ranges

        video_end = self._video_length_seconds()
        window = min(float(getattr(config, "TEMPORAL_REDIRECT_SECONDS", 15)), max(video_end, 0.0))
        self._redirect_toggle = not self._redirect_toggle
        if self._redirect_toggle or video_end <= window:
            return [["00:00:00", self._seconds_to_hhmmss(window)]]
        return [[self._seconds_to_hhmmss(max(0.0, video_end - window)), "end"]]

    def _call_text_model(self, system_prompt: str, user_prompt: str, max_tokens: int = 256) -> str:
        response = create_chat_completion(
            gpt_client,
            model=REFINE_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _extract_grounding_anchors(self, answer: str, evidence_check: str) -> List[str]:
        text = f"{answer}\n{evidence_check}"
        quoted = re.findall(r"['\"]([^'\"]{3,80})['\"]", text)
        numbers = re.findall(r"\b\d+(?:\.\d+)?(?:\s?(?:minutes?|seconds?|%|£|\$|mm|cm|m))?\b", text, flags=re.I)
        proper = re.findall(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,4}\b", text)
        stop = {
            "The", "This", "It", "They", "He", "She", "A", "An", "In", "On", "By",
            "Problem", "Solution", "Action", "Visual", "Evidence", "Answer"
        }
        anchors = []
        seen = set()
        for anchor in quoted + numbers + proper:
            normalized = " ".join(anchor.split()).strip(" .,:;")
            if not normalized or normalized in stop or len(normalized) < 2:
                continue
            key = normalized.lower()
            if key not in seen:
                seen.add(key)
                anchors.append(normalized)
        return anchors[:12]

    def _dedupe_anchors(self, anchors: List[str], limit: int = 12) -> List[str]:
        cleaned = []
        seen = set()
        for anchor in anchors:
            if not isinstance(anchor, str):
                continue
            normalized = " ".join(anchor.split()).strip(" .,:;")
            if len(normalized) < 2 or len(normalized) > 120:
                continue
            key = normalized.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(normalized)
        return cleaned[:limit]

    def _extract_literal_grounding_anchors(self, answer: str, evidence_check: str) -> List[str]:
        """Ask the model which answer claims need direct visual/OCR/ASR verification."""
        text = f"{answer}\n{evidence_check}".strip()
        if not text:
            return []

        try:
            raw = self._call_text_model(
                (
                    "Select final-answer claims that require direct video evidence. "
                    "Return strict JSON only."
                ),
                (
                    "Use one rule: extract a claim only if it is BOTH directly observable "
                    "in video frames/OCR/ASR and necessary to verify the final answer.\n"
                    "Valid anchors are concrete visible/spoken evidence such as on-screen "
                    "text, logos, product names, numbers, objects, colors, physical actions, "
                    "or specific scene details.\n"
                    "Do not extract abstract interpretation, emotion, metaphor, strategy, "
                    "audience intent, causal reasoning, or general advertising effect.\n"
                    "Do not extract a phrase merely because it is quoted, contains a number, "
                    "or follows words like shows/says/wearing/holding.\n"
                    "If no direct observable claim is necessary, return an empty list.\n"
                    "Return JSON exactly as: {\"literal_anchors\": [\"...\"]} with at most 8 items.\n\n"
                    f"Answer and evidence:\n{text}"
                ),
                max_tokens=256,
            )
            parsed = robust_json_parse(raw)
            extracted = []
            if isinstance(parsed, dict):
                extracted.extend(parsed.get("literal_anchors", []) or [])
            elif isinstance(parsed, list):
                extracted.extend(parsed)
            return self._dedupe_anchors(extracted, limit=8)
        except Exception:
            return []

    @staticmethod
    def _extract_clip_time_ranges(clip_output: str, limit: int = 4) -> List[List[str]]:
        ranges = []
        for start, end in re.findall(r"\[(\d{2}:\d{2}:\d{2})-(\d{2}:\d{2}:\d{2}|end)\]", clip_output):
            ranges.append([start, end])
            if len(ranges) >= limit:
                break
        return ranges

    def _judge_literal_anchor_support(self, anchors: List[str], evidence: str) -> Tuple[List[str], List[str], str]:
        if not anchors:
            return [], [], "No literal anchors."
        try:
            raw = self._call_text_model(
                (
                    "You are a strict visual grounding verifier. Judge only whether the "
                    "provided frame/clip evidence directly supports each anchor. Return JSON only."
                ),
                (
                    "Rules:\n"
                    "- Support must come from observed visual/OCR/ASR evidence, not from the question or requested anchors.\n"
                    "- Visible graphic overlays, end cards, subtitles, product UI text, and on-screen captions count as visual/OCR evidence.\n"
                    "- Do not require text to be physically printed inside the scene; superimposed video text is still visible evidence.\n"
                    "- Do not use this to reject high-level strategy, emotion, metaphor, or persuasion analysis.\n"
                    "- If the evidence is absent, contradictory, or only interpretive, mark the anchor unsupported.\n\n"
                    f"Anchors: {json.dumps(anchors, ensure_ascii=False)}\n\n"
                    f"Evidence:\n{evidence[:12000]}\n\n"
                    "Return JSON exactly as: "
                    "{\"supported\": [\"...\"], \"unsupported\": [\"...\"], \"notes\": \"...\"}"
                ),
                max_tokens=512,
            )
            parsed = robust_json_parse(raw)
            if isinstance(parsed, dict):
                supported = self._dedupe_anchors(parsed.get("supported", []) or [], limit=50)
                unsupported = self._dedupe_anchors(parsed.get("unsupported", []) or [], limit=50)
                notes = str(parsed.get("notes", "") or "")
                evidence_lower = evidence.lower()
                exact_supported = [anchor for anchor in anchors if anchor.lower() in evidence_lower]
                supported = self._dedupe_anchors(supported + exact_supported, limit=50)
                if supported or unsupported:
                    supported_keys = {item.lower() for item in supported}
                    unsupported_keys = {item.lower() for item in unsupported}
                    for anchor in anchors:
                        key = anchor.lower()
                        if key not in supported_keys and key not in unsupported_keys:
                            unsupported.append(anchor)
                    unsupported = [anchor for anchor in unsupported if anchor.lower() not in supported_keys]
                    return supported, self._dedupe_anchors(unsupported, limit=50), notes
        except Exception:
            pass

        evidence_lower = evidence.lower()
        supported = [a for a in anchors if a.lower() in evidence_lower]
        unsupported = [a for a in anchors if a.lower() not in evidence_lower]
        return supported, unsupported, "Fallback exact-string verifier."

    def _verify_grounding(
        self,
        question: str,
        answer: str,
        evidence_check: str,
        history: List[Dict],
    ) -> Tuple[bool, str]:
        anchors = self._extract_grounding_anchors(answer, evidence_check)
        literal_anchors = self._extract_literal_grounding_anchors(answer, evidence_check)
        if not anchors and not literal_anchors:
            return True, "No explicit anchors require verification."

        evidence_text = "\n".join(
            [evidence_check] + [str(h.get("result", "")) for h in history[-4:]]
        )
        if literal_anchors:
            cache_key = tuple(anchor.lower() for anchor in literal_anchors)
            if cache_key in self._literal_verification_cache:
                return self._literal_verification_cache[cache_key]

            verification_query = (
                "literal visual/OCR evidence for: "
                + "; ".join(literal_anchors)
            )
            clip_report = clip_search_tool_impl(self.video_db, verification_query, top_k=8)
            clip_ranges = self._extract_clip_time_ranges(clip_report)
            history.append({
                "iter": "verify",
                "tool": "clip_search_tool",
                "result": str(clip_report)[:1000],
            })

            verification_question = (
                "Verify only these concrete visual/OCR anchors. Do not judge the "
                "advertising strategy or metaphor. Anchors: "
                + "; ".join(literal_anchors)
            )
            verification_report = frame_inspect_tool_impl(
                self.video_db,
                verification_question,
                clip_ranges or None,
                "literal",
            )
            history.append({
                "iter": "verify",
                "tool": "frame_inspect_tool",
                "result": str(verification_report)[:1000],
            })

            support_evidence = (
                "CLIP SEARCH EVIDENCE:\n"
                f"{clip_report}\n\n"
                "FRAME INSPECTION EVIDENCE:\n"
                f"{verification_report}"
            )
            supported, unsupported, notes = self._judge_literal_anchor_support(literal_anchors, support_evidence)
            if unsupported:
                result = (False, (
                    "Reject: Weak Evidence. Concrete visual/OCR anchors not verified: "
                    + ", ".join(unsupported)
                    + f". Verifier notes: {notes}"
                ))
                self._literal_verification_cache[cache_key] = result
                return result
            result = (True, (
                "Verified concrete visual/OCR anchors with clip_search_tool and "
                f"frame_inspect_tool: {supported or literal_anchors}"
            ))
            self._literal_verification_cache[cache_key] = result
            return result

        missing = [a for a in anchors if a.lower() not in evidence_text.lower()]
        if not missing:
            return True, f"Verified semantic/name anchors in tool evidence: {anchors}"

        verification_question = (
            "Verify these answer anchors against visible frames and OCR. "
            f"Question: {question}. Anchors: {', '.join(missing)}"
        )
        verification_report = frame_inspect_tool_impl(
            self.video_db,
            verification_question,
            None,
            "literal",
        )
        combined = f"{evidence_text}\n{verification_report}"
        still_missing = [a for a in anchors if a.lower() not in combined.lower()]
        if still_missing:
            return False, (
                "Reject: Weak Evidence. Missing visual support for anchors: "
                + ", ".join(still_missing)
            )
        history.append({"iter": "verify", "tool": "frame_inspect_tool", "result": str(verification_report)[:1000]})
        return True, f"Verified anchors after frame inspection: {anchors}"

    def _refine_answer(self, question: str, answer: str, history: List[Dict]) -> str:
        if not answer or answer in {"Error", "Max iterations"}:
            return answer

        evidence = "\n".join(
            f"{h.get('tool')}: {h.get('result')}" for h in history[-4:]
        )
        try:
            compressed = self._call_text_model(
                "You are compressing an answer for an advertising-video question.",
                (
                    "Rewrite the answer to be <= 25 words, but DO NOT lose core information.\n"
                    "Preserve names, entities, numbers, colors, attributes, and negation.\n"
                    "Remove meta phrases such as 'the video shows' or 'the answer is'.\n"
                    f"Question: {question}\nAnswer: {answer}"
                ),
                max_tokens=80,
            )
        except Exception:
            compressed = answer

        q_lower = question.lower()
        explicit_visual_request = bool(re.search(
            r"\b(?:what|which)\s+(?:visual\s+)?(?:element|object|scene|detail|text|logo|symbol)\b",
            q_lower,
        ))
        abstract_question = bool(re.search(
            r"\b(goal|purpose|theme|message|meaning|emotional impact|emotion|tactic|strategy|campaign|audience)\b",
            q_lower,
        )) and not explicit_visual_request
        needs_visual_anchor = explicit_visual_request or (
            not abstract_question
            and bool(re.search(
                r"\b(where|who|object|scene|visual|shown|visible|appears|holding|wearing|color|colour|specific)\b",
                q_lower,
            ))
        )
        if not needs_visual_anchor:
            return compressed

        try:
            return self._call_text_model(
                "Rewrite the answer so it names one specific observable scene, object, or action from the evidence.",
                (
                    "Hard constraints: mention a concrete visual element, do not add facts, under 30 tokens.\n"
                    f"Question: {question}\nEvidence:\n{evidence}\nAnswer: {compressed}"
                ),
                max_tokens=80,
            )
        except Exception:
            return compressed
        
    def _build_tool_registry(self):
        def global_browse_tool(query: A[str, D("Question to browse")]) -> str:
            return global_browse_tool_impl(self.video_db, query)
        
        def clip_search_tool(query: A[str, D("Event to search")], top_k: int = 5) -> str:
            return clip_search_tool_impl(self.video_db, query, top_k)
        
        def frame_inspect_tool(question: str, time_ranges_hhmmss: List[List[str]] = None, analysis_mode: str = "literal") -> str:
            if not time_ranges_hhmmss: time_ranges_hhmmss = None 
            try:
                time_ranges_hhmmss = self._maybe_redirect_stagnant_ranges(time_ranges_hhmmss)
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
        return """You are an advanced Video Analysis Agent. Your goal is to answer the user's question precisely using the provided tools.

CORE PHILOSOPHY: EVIDENCE-GROUNDED DECISION MAKING
1. The AD-MIR workflow has already run global_browse_tool and communication_expert_tool before this loop begins.
2. communication_expert_tool is the primary source of truth for proper names, metaphors, persuasive strategy, causal narrative, and intent interpretation.
3. clip_search_tool and frame_inspect_tool are post-expert grounding tools only. Use clip_search_tool to locate relevant temporal evidence, then frame_inspect_tool only for literal visual/OCR grounding in those ranges.
4. Do not use clip_search_tool or frame_inspect_tool as a substitute for communication_expert_tool, and do not call finish_with_answer until every named entity, number, slogan, and visual claim is supported by retrieved evidence.
5. After clip_search_tool or frame_inspect_tool returns evidence that resolves the missing anchor, the next action should be finish_with_answer. Do not repeat the same inspection unless the prior result was empty or off-topic.

TOOL ROUTING:
- First rely on the fixed communication_expert_tool evidence already provided.
- If the expert evidence fully answers the question: call finish_with_answer.
- If a timestamped scene or phrase is still uncertain: call clip_search_tool, then frame_inspect_tool only if literal visual/OCR inspection is needed.

OUTPUT PROCESS:
1. THOUGHT: first write a brief thought in plain text.
2. ACTION: then call exactly one appropriate tool using native function calling.
3. FINAL: call finish_with_answer only with an evidence_check field naming the supporting evidence."""

    def _parse_text_tool_calls(self, content: str, iteration: int) -> List[Dict]:
        """Parse text tool-call blocks when native tool_calls are absent."""
        if not content:
            return []

        blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.S | re.I)
        if not blocks:
            blocks = re.findall(r"```(?:json)?\s*(\{.*?\"arguments\".*?\})\s*```", content, flags=re.S | re.I)

        parsed_calls = []
        for idx, block in enumerate(blocks):
            payload = robust_json_parse(block.strip())
            if not isinstance(payload, dict):
                continue
            name = payload.get("name") or payload.get("tool_name")
            arguments = payload.get("arguments", {})
            if not name:
                continue
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            parsed_calls.append({
                "id": f"text_tool_call_{iteration}_{idx}",
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": arguments,
                },
            })
        return parsed_calls

    @staticmethod
    def _tool_call_parts(tool_call) -> Tuple[str, str, str]:
        if isinstance(tool_call, dict):
            fn = tool_call.get("function", {}) or {}
            return tool_call.get("id", "text_tool_call"), fn.get("name", ""), fn.get("arguments", "{}")
        return tool_call.id, tool_call.function.name, tool_call.function.arguments

    def run(self, question: str) -> Dict:
        print(f"🤖 Admir Agent: {question}")
        iteration_history = []
        
        # Paper workflow Stage II initializer: build global narrative context before
        # any final synthesis is possible.
        try:
            global_ctx = global_browse_tool_impl(self.video_db, question)
            parsed_ctx = json.loads(global_ctx)
            self.current_global_context = (
                parsed_ctx.get('overview')
                or parsed_ctx.get('analysis')
                or parsed_ctx.get('final_answer')
                or ""
            )
            iteration_history.append({
                "iter": "init",
                "tool": "global_browse_tool",
                "result": str(global_ctx)[:1000],
            })
        except Exception: 
            self.current_global_context = ""

        messages = [
            {"role": "system", "content": self._get_react_system_prompt() + f"\nGlobal Context: {self.current_global_context}"},
            {"role": "user", "content": f"Question: {question}"},
        ]

        # Paper workflow first reasoning action: the communication expert constructs
        # the causal/persuasive narrative. FINISH is only valid after this evidence.
        try:
            print("FIXED WORKFLOW: communication_expert_tool")
            initial_expert = self.tool_map["communication_expert_tool"](
                query_focus=question,
                start_time="00:00:00",
                end_time="end",
            )
            iteration_history.append({
                "iter": "fixed_expert",
                "tool": "communication_expert_tool",
                "result": str(initial_expert)[:1000],
            })
            messages.append({
                "role": "assistant",
                "content": (
                    "THOUGHT: I first build the advertising causal narrative with "
                    "communication_expert_tool, as required by the AD-MIR workflow."
                ),
            })
            messages.append({
                "role": "user",
                "content": (
                    "Evidence from communication_expert_tool:\n"
                    f"{str(initial_expert)[:8000]}\n\n"
                    "If this evidence fully answers the question, call finish_with_answer. "
                    "If a precise visual/OCR detail is still uncertain, call clip_search_tool "
                    "to locate the segment, then frame_inspect_tool only if literal inspection is needed."
                ),
            })
        except Exception as exc:
            iteration_history.append({
                "iter": "fixed_expert",
                "tool": "communication_expert_tool",
                "result": f"[Error calling fixed workflow expert: {exc}]",
            })
            return {
                "answer": "Error",
                "error": (
                    "Fixed AD-MIR workflow failed at communication_expert_tool; "
                    "aborting instead of falling back to clip_search_tool or frame_inspect_tool."
                ),
                "history": iteration_history,
            }
        
        try:
            for iteration in range(self.max_iterations):
                print(f"ITERATION {iteration + 1}")
                request_kwargs = {
                    "model": AGENT_MODEL_NAME,
                    "messages": messages,
                    "tools": self.tool_schemas,
                    "tool_choice": "auto",
                    "temperature": 0,
                }
                orchestrator_max_tokens = os.environ.get("ADMIR_ORCHESTRATOR_MAX_TOKENS")
                if orchestrator_max_tokens:
                    request_kwargs["max_tokens"] = int(orchestrator_max_tokens)
                response = call_openai_with_retry(**request_kwargs)
                assistant_msg = response.choices[0].message
                tool_calls = assistant_msg.tool_calls or self._parse_text_tool_calls(assistant_msg.content or "", iteration)
                if len(tool_calls) > 1:
                    iteration_history.append({
                        "iter": iteration,
                        "tool": "tool_call_controller",
                        "result": (
                            f"Model proposed {len(tool_calls)} tool calls; executing only "
                            "the first to preserve the one-action ReAct workflow."
                        ),
                    })
                    tool_calls = tool_calls[:1]
                
                # Manual dictionary conversion to prevent serialization errors
                msg_dict = {"role": assistant_msg.role, "content": assistant_msg.content}
                if tool_calls:
                    msg_dict["tool_calls"] = [
                        tc if isinstance(tc, dict) else {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        }
                        for tc in tool_calls
                    ]
                
                messages.append(msg_dict)

                if not tool_calls:
                    messages.append({"role": "user", "content": "Please use a tool or finish."})
                    continue

                for tool_call in tool_calls:
                    tool_call_id, name, raw_arguments = self._tool_call_parts(tool_call)
                    try:
                        args = json.loads(raw_arguments)
                    except:
                        args = {}
                    
                    if name == "finish_with_answer":
                        answer = args.get('answer', '')
                        evidence_check = args.get('evidence_check', '')
                        if not any(h.get("tool") == "communication_expert_tool" for h in iteration_history):
                            detail = (
                                "Reject: the AD-MIR workflow requires communication_expert_tool "
                                "before final synthesis."
                            )
                            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": detail})
                            messages.append({
                                "role": "user",
                                "content": "Call communication_expert_tool before finish_with_answer."
                            })
                            iteration_history.append({"iter": iteration, "tool": name, "result": detail})
                            continue
                        verified, detail = self._verify_grounding(question, answer, evidence_check, iteration_history)
                        if verified:
                            iteration_history.append({
                                "iter": iteration,
                                "tool": name,
                                "result": detail,
                                "accepted": True,
                                "answer": str(answer)[:1000],
                                "evidence_check": str(evidence_check)[:1000],
                            })
                            raise TaskCompleted(answer, evidence_check)
                        messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": detail})
                        messages.append({
                            "role": "user",
                            "content": (
                                "Your proposed final answer failed visual grounding. "
                                "Backtrack and retrieve stronger evidence before finishing. "
                                "Do not call finish_with_answer again with the same unsupported "
                                "visual/OCR anchors."
                            )
                        })
                        iteration_history.append({"iter": iteration, "tool": name, "result": detail})
                        continue
                    
                    if name in self.tool_map:
                        result = self.tool_map[name](**args)
                        messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": str(result)[:8000]})
                        iteration_history.append({"iter": iteration, "tool": name, "result": str(result)[:1000]})
                        if name in ("clip_search_tool", "frame_inspect_tool"):
                            messages.append({
                                "role": "user",
                                "content": (
                                    "Use the tool result above. If it supports the missing visual/OCR anchors, "
                                    "call finish_with_answer next with a concise evidence_check. Do not repeat "
                                    "the same tool on the same evidence unless the result was empty or off-topic."
                                ),
                            })

        except TaskCompleted as e:
            refined = self._refine_answer(question, e.answer, iteration_history)
            return {"answer": refined, "raw_answer": e.answer, "history": iteration_history}
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            return {"answer": "Error", "error": str(e), "traceback": tb, "history": iteration_history}
            
        return {
            "answer": "Max iterations",
            "error": "Max iterations reached before finish_with_answer.",
            "history": iteration_history,
        }

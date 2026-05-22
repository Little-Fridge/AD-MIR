"""
Admir Build Database - Optimized
Hybrid Retrieval + Context-Anchored Subject Registry + Audio-Visual Fusion
"""
import os
import json
import glob
import re
import base64
from typing import Annotated as A, List, Tuple, Optional, Dict
import numpy as np
from nano_vectordb import NanoVectorDB
from tqdm import tqdm

import admir.config as config
from admir.func_call_schema import doc as D
from admir.utils import AzureOpenAIEmbeddingService, create_chat_completion

# Ensure necessary environment variables are set for underlying services.
if not os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = config.LOCAL_VLLM_BASE_URL

if getattr(config, "EMBEDDING_ENDPOINT", "") and "ADMIR_EMBEDDING_URL" not in os.environ:
    os.environ["ADMIR_EMBEDDING_URL"] = config.EMBEDDING_ENDPOINT

# =====================================================================
# Safety Limits
# =====================================================================
MAX_TOOL_OUTPUT_CHARS = 20000
MAX_SUBJECT_REGISTRY_CHARS = 1500
MAX_CAPTION_CHARS = 2000
MAX_VLM_RESPONSE_CHARS = 10000

def _require_strict_api_ready() -> None:
    validate = getattr(config, "validate_strict_paper_config", None)
    if callable(validate):
        validate(require_api_key=True)

def _truncate_output(text: str, max_length: int, label: str = "") -> str:
    """Safely truncate text to max length."""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length] + f"\n[TRUNCATED {label}]"

def _truncate_subject_registry(sr: dict, max_chars: int = MAX_SUBJECT_REGISTRY_CHARS) -> dict:
    """Truncate subject_registry to retain only key information."""
    if not sr or not isinstance(sr, dict):
        return sr
    
    sr_str = json.dumps(sr, ensure_ascii=False)
    if len(sr_str) <= max_chars:
        return sr
    
    # Retain summary of first 5 subjects
    truncated_sr = {}
    for i, (k, v) in enumerate(sr.items()):
        if i >= 5:
            break
        if isinstance(v, dict):
            truncated_sr[k] = {
                "name": v.get("name", "")[:100],
                "identity": v.get("identity", [])[:2] if isinstance(v.get("identity"), list) else []
            }
        else:
            truncated_sr[k] = str(v)[:100]
    
    truncated_sr["_note"] = f"Truncated from {len(sr)} subjects to 5"
    return truncated_sr

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# =====================================================================
# Helper Functions for Hybrid Search
# =====================================================================

def _extract_keywords(query: str) -> List[str]:
    """Extract keywords from query (removing stopwords)."""
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'can', 'could', 'may', 'might', 'this', 'that', 'these', 'those'
    }
    
    words = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return keywords

def _keyword_match_score(keywords: List[str], text: str) -> float:
    """Calculate keyword match score."""
    if not keywords or not text:
        return 0.0
    
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw in text_lower)
    return matches / len(keywords)

def _clip_semantic_affinity(a: Tuple, b: Tuple) -> float:
    try:
        va = np.array(a[4].get("__vector__", []), dtype=float)
        vb = np.array(b[4].get("__vector__", []), dtype=float)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return float(np.dot(va, vb) / denom) if denom else 0.0
    except Exception:
        return 0.0


def _merge_continuous_clips(
    clips: List[Tuple],
    threshold: float = 3.0,
    semantic_threshold: float = 0.8,
) -> List[Tuple]:
    """Merge clips with small temporal gaps or high semantic affinity."""
    if not clips:
        return []
    
    sorted_clips = sorted(clips, key=lambda x: x[0])
    
    merged = []
    current = list(sorted_clips[0])  # [start, end, caption, score, data]
    
    for clip in sorted_clips[1:]:
        gap_merge = clip[0] <= current[1] + threshold
        semantic_merge = _clip_semantic_affinity(tuple(current), clip) > semantic_threshold
        if gap_merge or semantic_merge:
            current[1] = max(current[1], clip[1])
            current[2] = current[2] + "\n" + clip[2]
            current[3] = max(current[3], clip[3])
        else:
            merged.append(tuple(current))
            current = list(clip)
    
    merged.append(tuple(current))
    return merged

# =====================================================================
# Subject Registry Embedding
# =====================================================================

_SUBJECT_EMBEDDINGS_CACHE = {}

def _compute_subject_embeddings(database: NanoVectorDB):
    """Compute embeddings for each subject in the registry."""
    additional_data = database.get_additional_data()
    subject_registry = additional_data.get('subject_registry', {})
    
    if not subject_registry or not isinstance(subject_registry, dict):
        return {}
    
    cache_key = str(database.storage_file) if hasattr(database, 'storage_file') else 'default'
    if cache_key in _SUBJECT_EMBEDDINGS_CACHE:
        return _SUBJECT_EMBEDDINGS_CACHE[cache_key]
    
    subject_embeddings = {}
    texts_to_embed = []
    subject_ids = []
    
    for subject_id, subject_info in subject_registry.items():
        if isinstance(subject_info, dict):
            parts = []
            if 'name' in subject_info:
                parts.append(f"Name: {subject_info['name']}")
            if 'appearance' in subject_info:
                parts.append(f"Appearance: {subject_info['appearance']}")
            if 'identity' in subject_info:
                identity = subject_info['identity']
                if isinstance(identity, list):
                    identity = ', '.join(identity)
                parts.append(f"Identity: {identity}")
            
            text = ' | '.join(parts)
            texts_to_embed.append(text)
            subject_ids.append(subject_id)
    
    if texts_to_embed:
        embeddings = AzureOpenAIEmbeddingService.get_embeddings(
            endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
            model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
            input_text=texts_to_embed,
            api_key=config.OPENAI_API_KEY,
        )
        
        if embeddings:
            for subject_id, emb_data in zip(subject_ids, embeddings):
                subject_embeddings[subject_id] = np.array(emb_data['embedding'])
    
    _SUBJECT_EMBEDDINGS_CACHE[cache_key] = subject_embeddings
    return subject_embeddings

def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def get_active_subject_registry(
    database: NanoVectorDB,
    query: str,
    global_context: str = "",
    top_k: int = 3,
) -> Dict:
    """Activate the subject registry with Q + global narrative as the semantic anchor."""
    additional_data = database.get_additional_data()
    subject_registry = additional_data.get("subject_registry", {})
    if not subject_registry or not isinstance(subject_registry, dict):
        return {}

    subject_embeddings = _compute_subject_embeddings(database)
    if not subject_embeddings:
        return _truncate_subject_registry(subject_registry)

    anchor = f"{query or ''}\n{global_context or ''}".strip()
    if not anchor:
        return _truncate_subject_registry(subject_registry)

    try:
        anchor_emb = AzureOpenAIEmbeddingService.get_embeddings(
            endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
            model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
            input_text=[anchor],
            api_key=config.OPENAI_API_KEY,
        )[0]["embedding"]
    except Exception as e:
        print(f"[WARN] Subject activation failed: {e}")
        return _truncate_subject_registry(subject_registry)

    anchor_vec = np.array(anchor_emb)
    ranked = []
    for subject_id, subject_vec in subject_embeddings.items():
        ranked.append((subject_id, _safe_cosine(np.array(subject_vec), anchor_vec)))
    ranked.sort(key=lambda x: x[1], reverse=True)

    active = {}
    for subject_id, score in ranked[:max(1, top_k)]:
        if subject_id in subject_registry:
            active[subject_id] = dict(subject_registry[subject_id])
        active[subject_id]["activation_score"] = round(score, 4)
    return active


def _caption_text_evidence(database: NanoVectorDB) -> str:
    """Collect explicit text snippets that captioning already saw."""
    snippets = []
    text_markers = ("text", "logo", "tagline", "slogan", "reads", "overlaid", "deliver in")
    for item in getattr(database, "_data", []):
        caption = str(item.get("caption", ""))
        if not caption or not any(marker in caption.lower() for marker in text_markers):
            continue
        snippets.extend(re.findall(r"['\"]([^'\"]{3,140})['\"]", caption))
        snippets.extend(re.findall(r"\b(?:WE\s+)?DELIVER(?:ED)?\s+IN\s+\d+\s+MINUTES\b", caption, flags=re.I))

    unique = []
    seen = set()
    for snippet in snippets:
        normalized = " ".join(str(snippet).split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return " | ".join(unique[:12])


def _delivery_minute_evidence(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"\b(?:we\s+)?deliver(?:ed)?\s+in\s+(\d+)\s+minutes?\b", re.I)
    return [(m.group(0), m.group(1)) for m in pattern.finditer(text or "")]


def _repair_ocr_text_with_caption_evidence(ocr_text: str, caption_text: str) -> str:
    """Surface OCR/caption conflicts instead of silently passing bad exact text."""
    if not ocr_text or not caption_text:
        return ocr_text

    ocr_minutes = _delivery_minute_evidence(ocr_text)
    caption_minutes = _delivery_minute_evidence(caption_text)
    if not ocr_minutes or not caption_minutes:
        return ocr_text

    ocr_numbers = {number for _, number in ocr_minutes}
    caption_numbers = {number for _, number in caption_minutes}
    if ocr_numbers == caption_numbers:
        return ocr_text

    caption_phrase = caption_minutes[0][0].upper()
    ocr_phrase = ocr_minutes[0][0].upper()
    warning = (
        "TEXT_CONFLICT_WARNING: OCR read "
        f"'{ocr_phrase}', while caption/log evidence reads '{caption_phrase}'. "
        "Use frame inspection or explicit visual evidence for exact numbers."
    )
    if warning in ocr_text:
        return ocr_text
    return f"{ocr_text} | CAPTION_TEXT_EVIDENCE: {caption_text} | {warning}"


def _get_multimodal_context(database: NanoVectorDB) -> Dict[str, str]:
    """Extract ASR context and metadata from the database."""
    additional_data = database.get_additional_data()
    
    context = {
        'asr': '',
        'asr_formatted': 'No audio transcript available.',
        'ocr': '',
        'ocr_formatted': '',
        'subject_registry': ''
    }
    
    # 1. ASR extraction
    asr_candidates = [
        additional_data.get('transcript'),
        additional_data.get('asr_text'),
        additional_data.get('asr_content'),
        additional_data.get('audio_transcript')
    ]
    
    asr_text = ""
    for candidate in asr_candidates:
        if candidate:
            if isinstance(candidate, str) and len(candidate.strip()) > 0:
                asr_text = candidate.strip()
                break
            elif isinstance(candidate, list):
                try:
                    parts = [str(x.get('text', x)) if isinstance(x, dict) else str(x) for x in candidate]
                    asr_text = " ".join(parts)
                    break
                except:
                    continue
    
    if asr_text:
        context['asr'] = asr_text
        context['asr_formatted'] = asr_text

    ocr_candidates = [
        additional_data.get('ocr_text'),
        additional_data.get('ocr_content'),
        additional_data.get('text_transcript')
    ]
    for candidate in ocr_candidates:
        if candidate:
            ocr_text = candidate if isinstance(candidate, str) else json.dumps(candidate, ensure_ascii=False)
            ocr_text = ocr_text.strip()
            if ocr_text:
                ocr_text = _repair_ocr_text_with_caption_evidence(ocr_text, _caption_text_evidence(database))
                context['ocr'] = ocr_text
                context['ocr_formatted'] = ocr_text
                break
    
    # 2. Subject Registry extraction
    if 'subject_registry' in additional_data:
        sr = additional_data['subject_registry']
        if sr:
            try:
                if isinstance(sr, (dict, list)):
                    sr_str = json.dumps(sr, ensure_ascii=False)
                else:
                    sr_str = str(sr)
                
                if len(sr_str) > 2000:
                    sr_str = sr_str[:2000] + "...[registry truncated]"
                
                context['subject_registry'] = sr_str
            except Exception as e:
                print(f"[WARN] Error formatting subject registry: {e}")
                context['subject_registry'] = ""
    
    return context

# =====================================================================
# Tools
# =====================================================================

def frame_inspect_tool(
    database: A[NanoVectorDB, D("The database containing video metadata.")],
    question: A[str, D("The specific detailed question to ask about the video content.")],
    time_ranges_hhmmss: A[list[tuple], D("Optional. Leave empty/None to scan the FULL video.")] = None, 
    analysis_mode: A[str, D("Optional: 'literal' for factual detection, or 'semantic' for analysis.")] = "literal"
) -> str:
    """Inspect video frames with Dynamic Sampling + Batch Processing."""
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"
    _require_strict_api_ready()
    from openai import OpenAI
    import httpx

    video_meta = database.get_additional_data()
    video_file_root = video_meta["video_file_root"]
    fps = float(video_meta.get("fps", getattr(config, "VIDEO_FPS", 1)) or 1)
    
    frames_dir = os.path.join(video_file_root, "frames")
    all_frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_n*.jpg")))
    
    if not all_frame_paths:
        return "Error: No frame files found in database directory."

    total_frames_available = len(all_frame_paths)
    selected_indices = []
    BATCH_SIZE = 5

    # [Branch A] GLOBAL SCAN
    if not time_ranges_hhmmss or len(time_ranges_hhmmss) == 0:
        idx_start = list(range(0, min(5, total_frames_available)))
        start_of_end_segment = max(0, total_frames_available - 5)
        idx_end = list(range(start_of_end_segment, total_frames_available))
        
        idx_middle = []
        if start_of_end_segment > 5:
            idx_middle = list(range(5, start_of_end_segment, 2))
            
        combined_set = set(idx_start + idx_middle + idx_end)
        selected_indices = sorted(list(combined_set))

    # [Branch B] FOCUSED SCAN
    else:
        def _ts_to_idx(ts: str) -> int:
            if str(ts).lower() == 'end': return total_frames_available - 1
            parts = str(ts).split(':')
            secs = 0.0
            if len(parts) == 3: secs = float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
            elif len(parts) == 2: secs = float(parts[0])*60 + float(parts[1])
            else: secs = float(ts)
            return int(round(secs * fps))

        raw_indices = set()
        for tr in time_ranges_hhmmss:
            try:
                s_idx = _ts_to_idx(tr[0])
                e_idx = _ts_to_idx(tr[1])
                s_idx = max(0, min(s_idx, total_frames_available - 1))
                e_idx = max(0, min(e_idx, total_frames_available - 1))
                if s_idx > e_idx: s_idx, e_idx = e_idx, s_idx 
                
                raw_indices.update(range(s_idx, e_idx + 1))
            except Exception: continue
        
        selected_indices = sorted(list(raw_indices))
        if len(selected_indices) > 90:
            idx_arr = np.linspace(0, len(selected_indices) - 1, 90, dtype=int)
            selected_indices = [selected_indices[i] for i in idx_arr]

    if not selected_indices:
        return "Error: Selection resulted in 0 frames."

    mm_context = _get_multimodal_context(database)
    asr_context = mm_context.get('asr_formatted', '')
    ocr_context = mm_context.get('ocr_formatted', '')
    active_subjects = get_active_subject_registry(database, question, top_k=3)
    active_subjects_text = json.dumps(active_subjects, ensure_ascii=False) if active_subjects else "None"
    
    base_instructions = ""
    if analysis_mode == 'literal':
        base_instructions = """
【MODE: LITERAL INSPECTION & OCR】
1. **VISUAL LOG**: Create a chronological log.
2. **TEXT**: Transcribe visible text/logos.
3. **DETAILS**: List objects and interactions.
"""
    else:
        base_instructions = """
【MODE: SEMANTIC & NARRATIVE】
1. **STORY**: Describe setup -> action -> outcome.
2. **TWIST**: Note environment changes.
3. **MEANING**: Identify symbols.
"""

    try:
        temp_client = OpenAI(
            base_url=config.LOCAL_VLLM_BASE_URL,
            api_key=config.OPENAI_API_KEY, 
            http_client=httpx.Client(timeout=250.0)
        )
    except Exception as e:
        return f"Error initializing OpenAI client: {str(e)}"

    chunks = [selected_indices[i:i + BATCH_SIZE] for i in range(0, len(selected_indices), BATCH_SIZE)]
    full_report = []
    
    print(f"  ...[Frame Inspect] Analyzing {len(selected_indices)} frames in {len(chunks)} batches...")

    for i, chunk_indices in enumerate(chunks):
        chunk_files = [all_frame_paths[idx] for idx in chunk_indices]
        batch_prompt = (
            f"{base_instructions}\n"
            f"Context (ASR): '{asr_context[:600]}...'\n"
            f"Context (OCR/Text): '{ocr_context[:600]}...'\n"
            f"Active subject registry: {active_subjects_text[:1200]}\n"
            f"User Question: {question}\n"
            f"--- BATCH INSTRUCTION ---\n"
            f"Part {i+1} of {len(chunks)}.\n"
        )

        content_list = [{"type": "text", "text": batch_prompt}]
        valid_count = 0
        for img_path in chunk_files:
            try:
                b64_str = encode_image_to_base64(img_path)
                if b64_str:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_str}", 
                            "detail": "high"
                        }
                    })
                    valid_count += 1
            except Exception: pass
        
        if valid_count == 0:
            continue

        try:
            response = create_chat_completion(
                temp_client,
                model=getattr(config, "AOAI_FRAME_INSPECT_MODEL_NAME", config.AOAI_TOOL_VLM_MODEL_NAME),
                messages=[{"role": "user", "content": content_list}], 
                max_tokens=int(os.environ.get("ADMIR_FRAME_INSPECT_MAX_TOKENS", "20000")), 
                temperature=0.0
            )
            full_report.append(f"--- [PART {i+1}] ---\n{response.choices[0].message.content}")
        except Exception as e:
            full_report.append(f"[Part {i+1} Failed: {str(e)}]")

    final_output = "\n".join(full_report)
    return _truncate_output(final_output, 15000, "merged_frame_inspect")

def clip_search_tool(
    database: A[NanoVectorDB, D("The database object.")],
    event_description: A[str, D("A textual description of the event.")],
    top_k: A[int, D("Number of results.")] = 5
) -> str:
    """Semantic Hybrid Search: LLM Rewrite + Vector Similarity + Keyword Boosting."""
    assert isinstance(database, NanoVectorDB), "Database error"
    _require_strict_api_ready()
    top_k = max(
        int(getattr(config, "CLIP_SEARCH_MIN_TOPK", 5)),
        min(int(top_k or 5), int(getattr(config, "OVERWRITE_CLIP_SEARCH_TOPK", 8))),
    )

    # [Step 0: Semantic Query Rewrite]
    def _rewrite_query(original_query: str) -> str:
        if len(original_query.split()) < 2 or '"' in original_query:
            return original_query
        try:
            _require_strict_api_ready()
            from openai import OpenAI
            import httpx
            exp_client = OpenAI(
                base_url=config.LOCAL_VLLM_BASE_URL,
                api_key=config.OPENAI_API_KEY, 
                http_client=httpx.Client(timeout=250.0)
            )
            
            prompt = (
                f"Rewrite '{original_query}' into 3 simple, comma-separated keywords/phrases for video retrieval.\n"
                "Rules: Keep it generic. Do NOT invent specific visual details "
                "(e.g. do NOT change 'luxury' to 'gold').\n"
                "Input: 'sad man' -> Output: crying person, unhappy face, depressed mood\n"
                f"Input: '{original_query}'\nOutput:"
            )
            resp = create_chat_completion(
                exp_client,
                model=config.AOAI_REFINE_LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.3
            )
            keywords = resp.choices[0].message.content.strip()
            return f"{original_query}. {keywords}"
        except Exception:
            return original_query

    search_query = _rewrite_query(event_description)

    # [Step 1: Expanded Vector Search]
    expanded_k = top_k * 3
    try:
        query_emb = AzureOpenAIEmbeddingService.get_embeddings(
            endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
            model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME, 
            input_text=[search_query],
            api_key=config.OPENAI_API_KEY,
        )[0]['embedding']
        results = database.query(query_emb, top_k=expanded_k)
    except Exception as e:
        return f"Search Error: {str(e)}"
    
    # [Step 2: Keyword Scoring]
    keywords = _extract_keywords(event_description)
    quoted_terms = re.findall(r'"([^"]*)"', event_description)
    scored_clips = []
    
    mm_context = _get_multimodal_context(database)
    asr_text_lower = mm_context.get('asr', '').lower()
    
    for data in results:
        caption = data['caption'].lower()
        base_score = data.get('__score__', 0.5) 
        cap_match_ratio = _keyword_match_score(keywords, caption)
        
        quote_boost = 0.0
        for q in quoted_terms:
            if q.lower() in caption:
                quote_boost += 3.0 
        
        asr_boost = 0.1 if any(k in asr_text_lower for k in keywords) else 0.0
        final_score = base_score + (cap_match_ratio * config.LEXICAL_MATCH_BETA) + quote_boost + asr_boost
        
        scored_clips.append((data['time_start_secs'], data['time_end_secs'], data['caption'], final_score, data))
    
    # [Step 3: Merge & Format]
    scored_clips.sort(key=lambda x: x[3], reverse=True)
    top_clips = scored_clips[:top_k]
    top_clips.sort(key=lambda x: x[0])
    merged_clips = _merge_continuous_clips(top_clips, threshold=3.0, semantic_threshold=0.8)
    
    def _sec_to_hhmmss(s: float) -> str:
        return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d}"

    lines = [f"[{_sec_to_hhmmss(c[0])}-{_sec_to_hhmmss(c[1])}] {c[2]}" for c in merged_clips]
    if not lines: return f"No relevant clips found for '{event_description}'."

    return "Relevant segments:\n\n" + _truncate_output("\n".join(lines), MAX_CAPTION_CHARS, "clip_search")


def global_browse_tool(
    database: A[NanoVectorDB, D("The database object.")],
    query: A[str, D("The user's question to contextualize.")],
) -> str:
    """Text-Based Forensic Reconstruction using Captions + ASR."""
    assert isinstance(database, NanoVectorDB), "Database error"
    _require_strict_api_ready()
    
    SWEEP_K = max(int(getattr(config, "GLOBAL_BROWSE_TOPK", 40)) * 4, 40)
    MAX_CONTEXT_CHARS = 40000 
    
    try:
        generic_query_emb = AzureOpenAIEmbeddingService.get_embeddings(
            endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
            model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
            input_text=["Detailed description of the video content"], 
            api_key=config.OPENAI_API_KEY,
        )[0]['embedding']
        candidates = database.query(generic_query_emb, top_k=SWEEP_K)
    except Exception as e:
        return json.dumps({"error": f"Retrieval failed: {str(e)}"})
    
    if not candidates:
        return json.dumps({"error": "No captions found in database."})

    candidates.sort(key=lambda x: x['time_start_secs'])
    browse_topk = int(getattr(config, "GLOBAL_BROWSE_TOPK", 40))
    if len(candidates) > browse_topk:
        boundary = max(1, min(5, browse_topk // 4))
        start_items = candidates[:boundary]
        end_items = candidates[-boundary:]
        middle = candidates[boundary:-boundary]
        remaining = browse_topk - len(start_items) - len(end_items)
        if remaining > 0 and middle:
            middle_indices = np.linspace(0, len(middle) - 1, remaining, dtype=int)
            middle_items = [middle[i] for i in middle_indices]
        else:
            middle_items = []
        candidates = sorted(start_items + middle_items + end_items, key=lambda x: x['time_start_secs'])
    
    log_entries = []
    for item in candidates:
        start_t = item['time_start_secs']
        t_str = f"[{int(start_t // 60):02d}:{int(start_t % 60):02d}]"
        raw_cap = item['caption'].strip()
        
        if "text" in raw_cap.lower() or "says" in raw_cap.lower() or '"' in raw_cap:
            log_entries.append(f"{t_str} [POTENTIAL_TEXT]: {raw_cap}")
        else:
            log_entries.append(f"{t_str} [SCENE]: {raw_cap}")

    visual_log = "\n".join(log_entries)
    mm_context = _get_multimodal_context(database)
    asr_transcript = mm_context.get('asr_formatted') or mm_context.get('asr', 'No audio transcript provided.')
    ocr_transcript = mm_context.get('ocr_formatted') or mm_context.get('ocr', 'No OCR text provided.')
    active_subjects = get_active_subject_registry(database, query, global_context=visual_log, top_k=3)
    active_subjects_text = json.dumps(active_subjects, ensure_ascii=False) if active_subjects else "None"

    system_prompt = """You are a Media Forensics Expert specialized in solving AdsQA cases.
**YOUR CONSTRAINT**: You cannot see the video. You only have:
1. Visual logs from clip captions.
2. Audio transcript.
3. OCR/text clues and active subject registry.
**STRATEGY FOR TEXT-ONLY ANALYSIS**:
1. Cross-reference visual and audio clues to infer product, brand, action, and intent.
2. Look specifically for [POTENTIAL_TEXT] tags and OCR clues.
3. Use active subjects to resolve pronouns and suppress background noise.
**OUTPUT FORMAT (JSON)**:
{
  "narrative_reconstruction": "Story flow (Hook -> Middle -> End).",
  "inferred_objects": ["List of objects"],
  "explicit_text_found": ["Text quoted in logs"],
  "audio_visual_mismatch": "Contradictions between seen and heard?",
  "final_answer": "Direct answer to USER QUERY."
}
"""
    user_content = (
        f"QUERY: {query}\n"
        f"AUDIO: {asr_transcript}\n"
        f"OCR/TEXT: {ocr_transcript}\n"
        f"ACTIVE SUBJECTS: {active_subjects_text}\n"
        f"LOGS: {visual_log[:MAX_CONTEXT_CHARS]}"
    )

    try:
        from openai import OpenAI
        import httpx
        temp_client = OpenAI(
            base_url=config.LOCAL_VLLM_BASE_URL, 
            api_key=config.OPENAI_API_KEY, 
            http_client=httpx.Client(timeout=300.0)
        )
        response = create_chat_completion(
            temp_client,
            model=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        return json.dumps({
            "analysis": response.choices[0].message.content,
            "evidence_source": "captions_and_asr"
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)})

# =====================================================================
# Initialization
# =====================================================================

def convert_seconds_to_hhmmss(seconds):
    total_seconds = float(seconds)
    hours = int(total_seconds // 3600)
    total_seconds %= 3600
    minutes = int(total_seconds // 60)
    secs = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def init_single_video_db(video_caption_json_path, output_video_db_path, emb_dim):
    vdb = NanoVectorDB(emb_dim, storage_file=output_video_db_path)
    if os.path.exists(output_video_db_path):
        print(f"Database {output_video_db_path} already exists.")
        if getattr(config, "STRICT_PAPER_MODE", False):
            with open(video_caption_json_path, "r", encoding="utf-8") as f:
                captions = json.load(f)
            expected_metadata = captions.get("_admir_metadata", {})
            actual_metadata = vdb.get_additional_data().get("caption_metadata", {})
            if actual_metadata != expected_metadata:
                raise RuntimeError(
                    f"Existing database {output_video_db_path} does not match "
                    "paper-strict caption metadata. Use a clean output root."
                )
    else:
        cap2emb_list = preprocess_captions(video_caption_json_path)
        data = []
        for idx, (timestamp, cap, emb) in enumerate(cap2emb_list):
            t1 = convert_seconds_to_hhmmss(timestamp[0])
            t2 = convert_seconds_to_hhmmss(timestamp[1])
            prefix = f"[From {t1} to {t2} seconds]\n"
            data.append(
                {
                    "__vector__": np.array(emb),
                    "time_start_secs": timestamp[0],
                    "time_end_secs": timestamp[1],
                    "caption": prefix + cap['caption'],
                }
            )
        _ = vdb.upsert(data)
        with open(video_caption_json_path, "r", encoding="utf-8") as f:
            captions = json.load(f)
        caption_metadata = captions.get('_admir_metadata', {})
        subject_registry = captions.get('subject_registry', captions.get('character_registry', None))
        clip_keys = [k for k in captions.keys() if re.match(r"^\d+(\.\d+)?_\d+(\.\d+)?$", str(k))]
        video_length = max([float(k.split("_")[1]) for k in clip_keys]) if clip_keys else 0
        video_length_str = convert_seconds_to_hhmmss(video_length)
        additional_data = {
            'subject_registry': subject_registry,
            'video_length': video_length_str,
            'video_file_root': os.path.dirname(os.path.dirname(video_caption_json_path)),
            'fps': getattr(config, "VIDEO_FPS", 2),
            'caption_metadata': caption_metadata,
        }
        vdb.store_additional_data(**additional_data)
        vdb.save()
    return vdb

def preprocess_captions(caption_json_path):
    with open(caption_json_path, "r", encoding="utf-8") as f:
        captions = json.load(f)
        
    scripts = []
    captions.pop('subject_registry', None)
    captions.pop('character_registry', None)
    captions.pop('_admir_metadata', None)
    
    for idx, (timestamp, cap_info) in enumerate(captions.items()):
        if not re.match(r"^\d+(\.\d+)?_\d+(\.\d+)?$", str(timestamp)):
            continue
        if isinstance(cap_info, str):
            cap_info = {"caption": cap_info}
        if cap_info.get('caption') is None or len(cap_info['caption']) == 0:
            continue
        elif isinstance(cap_info['caption'], list):
            cap_info['caption'] = cap_info['caption'][0]
        elif not isinstance(cap_info['caption'], str):
            cap_info['caption'] = str(cap_info['caption'])
            
        timestamp = list(map(float, timestamp.split("_")))
        scripts.append((timestamp, cap_info['caption'], cap_info))

    batch_size = 32
    batched_scripts = []
    for i in range(0, len(scripts), batch_size):
        batched_scripts.append(scripts[i:i+batch_size])
        
    cap2emb_list = []
    # Serial execution to avoid nested concurrency
    with tqdm(total=len(scripts), desc="Embedding captions...") as pbar:
        for batch in batched_scripts:
            result = single_batch_embedding_task(batch)
            cap2emb_list.extend(result)
            pbar.update(len(result))
            
    return cap2emb_list

def single_batch_embedding_task(data):
    timestamps, captions, cap_infos = map(list, (zip(*data)))
    embs = AzureOpenAIEmbeddingService.get_embeddings(
        endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
        model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
        input_text=captions,
        api_key=config.OPENAI_API_KEY,
    )
    max_tries = 3
    while embs is None or len(embs) != len(captions):
        max_tries -= 1
        if max_tries < 0:
            raise ValueError(f"Failed to get embeddings.")
        embs = AzureOpenAIEmbeddingService.get_embeddings(
            endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
            model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
            input_text=captions,
            api_key=config.OPENAI_API_KEY,
        )
    return list(zip(timestamps, cap_infos, [d['embedding'] for d in embs]))

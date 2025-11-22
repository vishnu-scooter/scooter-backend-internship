# main.py
import os
import uuid
import shutil
import subprocess
import functools
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import httpx
import numpy as np
import librosa

# Optional: semantic model (if installed). If not present, semantic similarity is skipped.
try:
    from sentence_transformers import SentenceTransformer, util
    EMB_AVAILABLE = True
except Exception:
    EMB_AVAILABLE = False
    SentenceTransformer = None
    util = None

# ---------------- Config / paths ----------------
BASE_DIR = Path("./data")
VIDEOS_DIR = BASE_DIR / "videos"
AUDIO_DIR = BASE_DIR / "audio"
CLIPS_DIR = BASE_DIR / "clips"

for p in [BASE_DIR, VIDEOS_DIR, AUDIO_DIR, CLIPS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

VAPI_BASE_URL = os.getenv("VAPI_BASE_URL", "").rstrip("/")  # e.g. https://api.vapi.ai
VAPI_API_KEY = os.getenv("VAPI_API_KEY", "")  # Bearer token or API key
FFMPEG_PATH = shutil.which("ffmpeg")  # must be available on PATH

# logger
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Interview Highlight Pipeline (VAPI-based)")

# Simple in-memory DB for demo: {video_id: {...}}
DB: Dict[str, Dict[str, Any]] = {}

# Keyword categories (customize as needed)
KEYWORD_CATEGORIES = {
    "leadership": ["lead", "led", "manage", "supervise", "coordinate", "mentor", "guide", "initiative"],
    "achievement": ["achieved", "improved", "delivered", "growth", "success", "accomplished"],
    "problem_solving": ["solve", "resolved", "debugged", "fixed", "tackle", "analyzed"],
    "technical": ["implemented", "designed", "developed", "engineer", "build", "optimized"],
    "soft_skills": ["communicate", "communication", "teamwork", "collaboration", "presentation", "confidence", "empathy"]
}
ACTION_VERBS = ["led","improved","resolved","implemented","designed","debugged","optimized","collaborated","achieved","managed","created","initiated","built","developed"]

# Optional semantic model (loaded lazily)
EMB_MODEL = None

# ---------------- Helpers ----------------

def ensure_ffmpeg():
    if FFMPEG_PATH is None:
        raise RuntimeError("ffmpeg not found on PATH. Please install ffmpeg and ensure it's available.")

def run_subprocess(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True)

async def run_blocking(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))

# ---------------- ffmpeg audio extraction & cutting ----------------

def extract_audio_ffmpeg_sync(video_path: Path, out_wav: Path) -> Path:
    ensure_ffmpeg()
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        str(out_wav)
    ]
    proc = run_subprocess(cmd)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors='ignore')
        raise RuntimeError(f"ffmpeg error extracting audio: {stderr}")
    return out_wav

def cut_video_segment_sync(video_path: Path, start: float, end: float, out_path: Path) -> Path:
    ensure_ffmpeg()
    duration = max(0.01, end - start)
    cmd = [
        "ffmpeg", "-y", "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration), "-c", "copy", str(out_path)
    ]
    proc = run_subprocess(cmd)
    if proc.returncode != 0:
        # fallback re-encode
        cmd2 = [
            "ffmpeg", "-y", "-ss", str(start), "-i", str(video_path),
            "-t", str(duration), "-c:v", "libx264", "-c:a", "aac", str(out_path)
        ]
        proc2 = run_subprocess(cmd2)
        if proc2.returncode != 0:
            stderr = proc2.stderr.decode(errors='ignore')
            raise RuntimeError(f"ffmpeg cut error: {stderr}")
    return out_path

# ---------------- VAPI transcript fetching ----------------

async def get_vapi_transcript(vapi_call_id: str) -> List[Dict[str,Any]]:
    """
    Fetch transcript JSON array from VAPI for the given call id.
    Expect VAPI to return JSON array of segments, each with:
      - speaker (string)
      - text (string)
      - start (float seconds)
      - end (float seconds)
    If your VAPI URL path differs, edit this function accordingly.
    """
    if not VAPI_BASE_URL or not VAPI_API_KEY:
        raise RuntimeError("VAPI_BASE_URL and VAPI_API_KEY must be set in environment variables.")

    # Best-effort default path — modify if your actual endpoint differs
    url = f"{VAPI_BASE_URL}/calls/{vapi_call_id}/transcript"

    headers = {"Authorization": f"Bearer {VAPI_API_KEY}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch transcript from VAPI: {r.status_code} - {r.text}")
    data = r.json()
    # If VAPI returns wrapper object, try to extract array
    if isinstance(data, dict):
        # try common keys
        for k in ("transcript","segments","utterances","messages"):
            if k in data and isinstance(data[k], list):
                return data[k]
        # maybe the object itself is single-segment array
        raise RuntimeError("VAPI transcript response is JSON object; could not find transcript array. Response keys: " + ", ".join(list(data.keys())))
    elif isinstance(data, list):
        return data
    else:
        raise RuntimeError("Unexpected transcript format from VAPI")

# ---------------- audio analysis helpers ----------------

def detect_first_voice_after_sync(wav_path: Path, window_start: float, window_end: float, rms_threshold: float = 0.01, hop_length: int = 512, frame_length: int = 2048) -> Optional[float]:
    """
    Return timestamp of first frame where RMS > threshold between window_start and window_end.
    Returns None if not found.
    """
    try:
        y, sr = librosa.load(str(wav_path), sr=16000, offset=window_start, duration=max(0.01, window_end - window_start))
    except Exception:
        return None
    if y.size == 0:
        return None
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    for i, v in enumerate(rms):
        if v > rms_threshold:
            # timestamp relative to file
            t = window_start + (i * hop_length) / sr
            return float(t)
    return None

def compute_audio_features_sync(wav_path: Path, start: float, end: float):
    try:
        y, sr = librosa.load(str(wav_path), sr=16000, offset=start, duration=max(0.01, end - start))
    except Exception:
        return {"rms": 0.0, "speech_rate": 0.0}
    if y.size == 0:
        return {"rms": 0.0, "speech_rate": 0.0}
    rms = float(np.mean(librosa.feature.rms(y=y)))
    words_est = max(1, int(len(y) / (sr * 0.3)))
    speech_rate = words_est / max(0.1, end - start)
    return {"rms": rms, "speech_rate": speech_rate}

# ---------------- scoring ----------------

def score_segment_text_sync(text: str, job_desc: str = "") -> Dict[str,Any]:
    text_l = (text or "").lower()
    matched_keywords = []
    action_hits = []
    total_keyword_score = 0.0
    category_scores = {}
    for cat, words in KEYWORD_CATEGORIES.items():
        cat_score = 0.0
        for w in words:
            if w in text_l:
                matched_keywords.append(w)
                cat_score += 1.0
        if cat_score > 0:
            category_scores[cat] = cat_score
            total_keyword_score += cat_score
    keyword_score = min(total_keyword_score / 10.0, 1.0)
    for v in ACTION_VERBS:
        if v in text_l:
            action_hits.append(v)
    action_score = min(len(action_hits) * 0.1, 1.0)
    sem_sim = 0.0
    if job_desc and EMB_AVAILABLE:
        try:
            global EMB_MODEL
            if EMB_MODEL is None:
                EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            emb1 = EMB_MODEL.encode(text, convert_to_tensor=True)
            emb2 = EMB_MODEL.encode(job_desc, convert_to_tensor=True)
            sem_sim = float(util.cos_sim(emb1, emb2).item())
            sem_sim = (sem_sim + 1) / 2
        except Exception:
            sem_sim = 0.0
    soft_skill_bonus = 0.1 if any(sk in text_l for sk in KEYWORD_CATEGORIES["soft_skills"]) else 0.0
    text_score = 0.5 * sem_sim + 0.3 * keyword_score + 0.1 * action_score + 0.1 * soft_skill_bonus
    return {
        "semantic_similarity": sem_sim,
        "keyword_score": keyword_score,
        "action_score": action_score,
        "soft_skill_bonus": soft_skill_bonus,
        "matched_keywords": matched_keywords,
        "action_verbs": action_hits,
        "category_scores": category_scores,
        "text_score": min(text_score, 1.0)
    }

# ---------------- speaker detection helper ----------------

def is_candidate_speaker_label(s: str) -> bool:
    if not s:
        return False
    s_low = s.strip().lower()
    # common labels from VAPI might be "user", "agent", "customer", "interviewee", etc.
    return s_low in ("user", "candidate", "interviewee", "participant", "customer", "caller", "human")

# ---------------- pipeline ----------------

async def run_pipeline_for_video(video_id: str, vapi_call_id: str, job_desc: str = "") -> Dict[str,Any]:
    entry = DB.get(video_id)
    if not entry:
        raise ValueError("video_id not found")
    video_path = Path(entry["video_path"])
    wav_path = AUDIO_DIR / f"{video_id}.wav"

    # extract audio
    await run_blocking(extract_audio_ffmpeg_sync, video_path, wav_path)

    # fetch transcript from VAPI
    transcript = await get_vapi_transcript(vapi_call_id)
    # transcript is expected to be list of segments with speaker,text,start,end
    if not isinstance(transcript, list):
        raise RuntimeError("VAPI transcript parse error: expected JSON array of segments")

    # Filter & keep candidate segments (speaker matches candidate)
    candidate_segments = []
    for seg in transcript:
        # support a variety of field names / shapes
        speaker = seg.get("speaker") or seg.get("role") or seg.get("who") or ""
        text = seg.get("text") or seg.get("content") or ""
        # timestamp fields - try multiple common keys
        start = seg.get("start") if seg.get("start") is not None else seg.get("offset") if seg.get("offset") is not None else seg.get("from", None)
        end = seg.get("end") if seg.get("end") is not None else seg.get("end_time") if seg.get("end_time") is not None else seg.get("to", None)
        # If timestamps are missing or strings, try to coerce
        try:
            start = float(start) if start is not None else None
            end = float(end) if end is not None else None
        except Exception:
            start = None
            end = None
        if start is None or end is None:
            # skip segments with no usable timestamps
            continue
        if is_candidate_speaker_label(speaker):
            candidate_segments.append({"start": float(start), "end": float(end), "text": text, "speaker": speaker})

    # For each candidate segment, find true answer start (detect first voice after start)
    qa_items = []
    for seg in candidate_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        detected_start = await run_blocking(detect_first_voice_after_sync, wav_path, seg_start, seg_end, 0.01)
        if detected_start is None:
            answer_start = seg_start
        else:
            # Use detected voice timestamp, with a small left-trim tolerance (e.g. 0.05s)
            answer_start = max(seg_start, detected_start)
        # guard
        if answer_start >= seg_end:
            answer_start = seg_start
        qa_items.append({
            "question": None,  # VAPI may include question segments labeled 'agent' — user may stitch them separately
            "answer_start": float(answer_start),
            "answer_end": float(seg_end),
            "answer_text": seg.get("text", "")
        })

    # Score and produce highlights
    highlights = []
    for item in qa_items:
        s = item["answer_start"]
        e = item["answer_end"]
        text = item["answer_text"]
        audio_feats = await run_blocking(compute_audio_features_sync, wav_path, s, e)
        text_score = await run_blocking(score_segment_text_sync, text, job_desc)
        composite = float(text_score["text_score"] * 0.7 + audio_feats["rms"] * 0.2 + audio_feats["speech_rate"] * 0.1)
        highlights.append({
            "start": s,
            "end": e,
            "answer": text,
            "scores": text_score,
            "audio_features": audio_feats,
            "score": composite
        })

    # sort and keep top 3
    highlights_sorted = sorted(highlights, key=lambda x: x["score"], reverse=True)[:3]

    # cut clips
    clip_infos = []
    for i, h in enumerate(highlights_sorted, start=1):
        clip_path = CLIPS_DIR / f"{video_id}_highlight_{i}.mp4"
        await run_blocking(cut_video_segment_sync, video_path, h["start"], h["end"], clip_path)
        clip_infos.append({
            "clip_path": str(clip_path),
            "start": h["start"],
            "end": h["end"],
            "score": h["score"],
            "answer": h["answer"]
        })

    entry["highlights"] = clip_infos
    return {"video_id": video_id, "highlights": clip_infos}

# ---------------- API endpoints ----------------

class UploadResponse(BaseModel):
    video_id: str
    filename: str

@app.post("/upload-video", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported video format")
    vid = str(uuid.uuid4())
    dest = VIDEOS_DIR / f"{vid}_{file.filename}"
    # save file (blocking) in threadpool
    await run_blocking(shutil.copyfileobj, file.file, open(dest, "wb"))
    DB[vid] = {"video_path": str(dest), "filename": file.filename}
    return {"video_id": vid, "filename": file.filename}

@app.post("/process/{video_id}")
async def process_video(video_id: str, vapi_call_id: str, job_desc: str = ""):
    """
    Process uploaded video using VAPI transcript for call id = vapi_call_id.
    Example: POST /process/{video_id}?vapi_call_id=CALL123&job_desc=backend
    """
    try:
        res = await run_pipeline_for_video(video_id, vapi_call_id, job_desc)
        return JSONResponse(content=res)
    except Exception as e:
        logger.exception("Processing error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/highlights/{video_id}")
async def get_highlights(video_id: str):
    entry = DB.get(video_id)
    if not entry:
        raise HTTPException(status_code=404, detail="video not found")
    return entry.get("highlights", [])

@app.get("/download-clip/{video_id}/{index}")
async def download_clip(video_id: str, index: int):
    entry = DB.get(video_id)
    if not entry:
        raise HTTPException(status_code=404, detail="video not found")
    highlights = entry.get("highlights")
    if not highlights or index < 1 or index > len(highlights):
        raise HTTPException(status_code=404, detail="clip not found")
    clip_path = highlights[index - 1]["clip_path"]
    if not Path(clip_path).exists():
        raise HTTPException(status_code=404, detail="clip missing on disk")
    return FileResponse(clip_path, media_type="video/mp4", filename=Path(clip_path).name)

# ---------------- Startup checks ----------------

@app.on_event("startup")
async def startup_checks():
    # log ffmpeg presence
    if FFMPEG_PATH is None:
        logger.warning("ffmpeg not found; ffmpeg-dependent endpoints will fail.")
    else:
        logger.info(f"ffmpeg found at {FFMPEG_PATH}")
    # try lazy load embedding model only if env var requests it
    if EMB_AVAILABLE and os.getenv("LOAD_EMBEDDING_MODEL", "false").lower() in ("1","true","yes"):
        try:
            global EMB_MODEL
            EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded embedding model at startup.")
        except Exception as ex:
            logger.warning("Could not load embedding model at startup: %s", ex)

if __name__ == "__main__":
    print("Run: uvicorn main:app --reload --port 8000")
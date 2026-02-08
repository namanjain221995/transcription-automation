"""
test6.py — Drive-integrated Candidate Eye/Face Tracking (slot-based)
FAST BY DEFAULT (just run: python test6.py)

✅ DEFAULT BEHAVIOR (no flags needed):
- Uploads:
  - __EYE_result.json
  - __EYE_annotated_h264.mp4  (annotated video ON)
- Annotated video shows FULL FACE MESH (like your screenshot) + eye/iris dots/boxes + LOOKING/Focus label
- STRICT_REF_ONLY=True:
  If reference image exists and REF_SIM < ref_min_sim OR no match -> treat as TRAINER FRAME:
    - Do NOT run FaceMesh
    - Do NOT track gaze
    - Do NOT draw mesh
- Do NOT upload:
  - __EYE_summary.json
  - __EYE_metrics.csv
- Do NOT process any input video that contains '__EYE_'

🚀 SPEED OPTIMIZATIONS (CPU/RAM heavy):
- ROI-first InsightFace matching (huge speed-up)
- Periodic full-frame reacquire to recover candidate movement
- Uses all CPU threads (OpenMP/MKL/NumExpr + OpenCV threads)
- Downscale for processing by default (FAST_DEFAULT_DOWNSCALE), annotated output uses that resolution

Install:
  pip install opencv-python mediapipe numpy pandas python-dotenv openai insightface onnxruntime

Requires:
  - ffmpeg installed and in PATH
  - credentials.json next to script
  - token.json created on first run
  - OPENAI_API_KEY in .env

Optional ENV:
  SLOT_CHOICE=2
  USE_SHARED_DRIVES=1
  HEADLESS_AUTH=1
  OPENAI_MODEL=gpt-5
  MAX_VIDEO_RETRIES=3
  VIDEO_RETRY_BASE_SLEEP=5
"""

import os

# ==========================================================
# MAX CPU/RAM UTILIZATION (set BEFORE numpy/onnx/insightface)
# ==========================================================
_CPU = os.cpu_count() or 8
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_CPU))
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import io
import re
import json
import time
import random
import tempfile
import subprocess
import argparse
from pathlib import Path
from dataclasses import dataclass
from math import sqrt
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, Counter

import cv2
import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# InsightFace
from insightface.app import FaceAnalysis

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import mediapipe as mp


# =========================
# ENV
# =========================
SLOT_CHOICE = (os.getenv("SLOT_CHOICE") or "").strip()
USE_SHARED_DRIVES = (os.getenv("USE_SHARED_DRIVES") or "").strip().lower() in ("1", "true", "yes", "y")
DEFAULT_OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "").strip() or "gpt-5"
HEADLESS_AUTH = (os.getenv("HEADLESS_AUTH") or "").strip().lower() in ("1", "true", "yes", "y")

# =========================
# FAST DEFAULTS (no flags needed)
# =========================
FAST_DEFAULT_MATCH_EVERY_N = int((os.getenv("FAST_MATCH_EVERY_N") or "60").strip() or 60)

# Default processing resolution (annotated output uses this resolution).
# Recommended:
# - If your originals are 1080p → 1280x720 is a great compromise (fast + good quality).
# - If your originals are 720p → you can set FAST_DEFAULT_DOWNSCALE="" to keep original resolution.
FAST_DEFAULT_DOWNSCALE = (os.getenv("FAST_DOWNSCALE") or "1280x720").strip()

# ROI matching tuning
ROI_PAD = float((os.getenv("ROI_PAD") or "0.55").strip() or 0.55)
FULL_FRAME_EVERY_K_MATCHES = int((os.getenv("FULL_FRAME_EVERY_K_MATCHES") or "6").strip() or 6)
MAX_STALE_MULT = float((os.getenv("MAX_STALE_MULT") or "2.0").strip() or 2.0)

# =========================
# VIDEO RETRIES (per video)
# =========================
MAX_VIDEO_RETRIES = int((os.getenv("MAX_VIDEO_RETRIES") or "3").strip() or 3)
VIDEO_RETRY_BASE_SLEEP = float((os.getenv("VIDEO_RETRY_BASE_SLEEP") or "5").strip() or 5.0)


def is_retryable_video_error(e: Exception) -> bool:
    if isinstance(e, (BrokenPipeError, ConnectionError, TimeoutError)):
        return True
    msg = f"{type(e).__name__}: {e}".lower()
    retry_keywords = [
        "broken pipe",
        "connection reset",
        "connection aborted",
        "timed out",
        "temporarily unavailable",
        "internal error",
        "backend error",
        "rate limit",
        "quota",
        "429",
        "500",
        "502",
        "503",
        "504",
        "moov atom not found",
        "invalid data found",
        "error while decoding",
        "ffmpeg",
    ]
    return any(k in msg for k in retry_keywords)


# =========================
# CONFIG (Drive)
# =========================
SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE = Path("token.json")
FOLDER_MIME = "application/vnd.google-apps.folder"

ROOT_FOLDER_NAME = "2025"  # your Drive root folder name

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

FOLDER_NAMES_TO_PROCESS = [
    "3. Introduction Video",
    "5. Project Scenarios",
    "8. Tools & Technology Videos",
]

SKIP_PERSON_FOLDERS = {"1. Format"}

CANDIDATE_IMAGE_NAME_RX = re.compile(
    r"(candidate|profile|photo|face|selfie|headshot).*?\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)

ANNOT_SUFFIX = "__EYE_annotated_h264.mp4"
SUMMARY_SUFFIX = "__EYE_summary.json"   # local-only
RESULT_SUFFIX = "__EYE_result.json"     # upload
METRICS_SUFFIX = "__EYE_metrics.csv"    # local-only

# ✅ STRICT RULE
STRICT_REF_ONLY = True


# =========================
# RETRIES (Drive)
# =========================
def execute_with_retries(request, *, max_retries: int = 8, base_sleep: float = 1.0):
    for attempt in range(max_retries):
        try:
            return request.execute()
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status in (429, 500, 502, 503, 504):
                if attempt == max_retries - 1:
                    raise
                sleep = (base_sleep * (2 ** attempt)) + random.random()
                print(f"[WARN] Drive transient HTTP {status}. Retry in {sleep:.1f}s")
                time.sleep(sleep)
                continue
            raise


# =========================
# Shared Drives kwargs
# =========================
def _kwargs_for_list() -> Dict[str, Any]:
    if USE_SHARED_DRIVES:
        return {
            "supportsAllDrives": True,
            "includeItemsFromAllDrives": True,
            "corpora": "allDrives",
        }
    return {}


def _kwargs_for_mutation() -> Dict[str, Any]:
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True}
    return {}


def _get_media_kwargs() -> Dict[str, Any]:
    if USE_SHARED_DRIVES:
        return {"supportsAllDrives": True}
    return {}


# =========================
# Google Drive Auth
# =========================
def get_drive_service():
    creds: Optional[Credentials] = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if creds and set(creds.scopes or []) != set(SCOPES):
        print("[AUTH] token.json scopes mismatch. Deleting token.json and re-authenticating...")
        TOKEN_FILE.unlink(missing_ok=True)
        creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"[AUTH] Refresh failed ({e}). Re-authenticating...")
                TOKEN_FILE.unlink(missing_ok=True)
                creds = None

        if not creds or not creds.valid:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError("credentials.json not found next to this script.")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)

            if HEADLESS_AUTH or not os.environ.get("DISPLAY"):
                creds = flow.run_console()
            else:
                creds = flow.run_local_server(port=0)

            TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds)


# =========================
# Drive Helpers
# =========================
def _escape_drive_q_value(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def drive_search_folder_anywhere(service, folder_name: str) -> List[dict]:
    safe = _escape_drive_q_value(folder_name)
    q = f"name = '{safe}' and mimeType = '{FOLDER_MIME}' and trashed=false"

    out: List[dict] = []
    page_token = None
    while True:
        res = execute_with_retries(
            service.files().list(
                q=q,
                fields="nextPageToken, files(id,name,parents,modifiedTime)",
                pageSize=1000,
                pageToken=page_token,
                **_kwargs_for_list(),
            )
        )
        out.extend(res.get("files", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return out


def pick_best_named_folder(candidates: List[dict]) -> dict:
    return sorted(candidates, key=lambda c: c.get("modifiedTime") or "", reverse=True)[0]


def drive_list_children(service, parent_id: str, mime_type: Optional[str] = None):
    q_parts = [f"'{parent_id}' in parents", "trashed = false"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    page_token = None
    while True:
        res = execute_with_retries(
            service.files().list(
                q=q,
                fields="nextPageToken, files(id,name,mimeType,modifiedTime,size)",
                pageSize=1000,
                pageToken=page_token,
                **_kwargs_for_list(),
            )
        )
        for f in res.get("files", []):
            yield f
        page_token = res.get("nextPageToken")
        if not page_token:
            break


def drive_find_child(service, parent_id: str, name: str, mime_type: Optional[str] = None) -> Optional[dict]:
    safe = _escape_drive_q_value(name)
    q_parts = [f"'{parent_id}' in parents", "trashed = false", f"name = '{safe}'"]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")
    q = " and ".join(q_parts)

    res = execute_with_retries(
        service.files().list(
            q=q,
            fields="files(id,name,mimeType,parents,modifiedTime)",
            pageSize=50,
            **_kwargs_for_list(),
        )
    )
    files = res.get("files", []) or []
    if not files:
        return None
    return sorted(files, key=lambda f: f.get("modifiedTime") or "", reverse=True)[0]


def drive_download_file(service, file_id: str, dest_path: Path):
    request = service.files().get_media(fileId=file_id, **_get_media_kwargs())
    with io.FileIO(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024 * 8)
        done = False
        while not done:
            try:
                _, done = downloader.next_chunk()
            except HttpError as e:
                status = getattr(e.resp, "status", None)
                if status in (429, 500, 502, 503, 504):
                    sleep = 1.0 + random.random()
                    print(f"[WARN] Download transient HTTP {status}. Retry in {sleep:.1f}s")
                    time.sleep(sleep)
                    continue
                raise


def drive_upload_file(service, parent_id: str, drive_filename: str, local_path: Path, mime_type: str):
    existing = drive_find_child(service, parent_id, drive_filename, None)
    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)

    if existing:
        execute_with_retries(
            service.files().update(
                fileId=existing["id"],
                media_body=media,
                **_kwargs_for_mutation(),
            )
        )
        return existing["id"]
    else:
        meta = {"name": drive_filename, "parents": [parent_id]}
        created = execute_with_retries(
            service.files().create(
                body=meta,
                media_body=media,
                fields="id",
                **_kwargs_for_mutation(),
            )
        )
        return created["id"]


# =========================
# SLOT SELECTION
# =========================
def list_slot_folders(service, slots_parent_id: str) -> List[dict]:
    return sorted(
        list(drive_list_children(service, slots_parent_id, FOLDER_MIME)),
        key=lambda x: (x.get("name") or "").lower(),
    )


def choose_slot(service, slots_parent_id: str) -> dict:
    slots = list_slot_folders(service, slots_parent_id)
    if not slots:
        raise RuntimeError("No slot folders found under ROOT folder.")

    if SLOT_CHOICE.isdigit():
        idx = int(SLOT_CHOICE)
        if 1 <= idx <= len(slots):
            chosen = slots[idx - 1]
            print(f"[AUTO] Using SLOT_CHOICE={idx}: {chosen['name']}")
            return chosen
        raise RuntimeError(f"SLOT_CHOICE='{SLOT_CHOICE}' out of range (1..{len(slots)}).")

    print("\n" + "=" * 80)
    print("SELECT SLOT TO PROCESS")
    print("=" * 80)
    for i, s in enumerate(slots, start=1):
        print(f"  {i:2}. {s['name']}")
    print("  EXIT - Exit\n")

    while True:
        choice = input("Choose slot number (e.g. 1) or EXIT: ").strip().lower()
        if choice == "exit":
            raise SystemExit(0)
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(slots):
                return slots[idx - 1]
        print(" Invalid choice. Try again.")


# =========================
# Utilities
# =========================
def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def eye_aspect_ratio(pts: List[Tuple[float, float]]) -> float:
    p1, p2, p3, p4, p5, p6 = pts
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-6)


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def put_text_with_bg(img, text, org, font_scale=1.0, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad = 6
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def robust_center(samples: List[float]) -> float:
    if len(samples) < 10:
        return 0.5
    a = np.array(samples, dtype=np.float32)
    q1, q3 = np.percentile(a, [25, 75])
    iqr = max(q3 - q1, 1e-6)
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    a2 = a[(a >= lo) & (a <= hi)]
    if len(a2) < 5:
        return float(np.median(a))
    return float(np.median(a2))


def rotationMatrixToEulerAngles(R: np.ndarray) -> Tuple[float, float, float]:
    sy = sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])  # pitch
        y = np.arctan2(-R[2, 0], sy)      # yaw
        z = np.arctan2(R[1, 0], R[0, 0])  # roll
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return (float(np.degrees(x)), float(np.degrees(y)), float(np.degrees(z)))


# =========================
# FaceMesh landmark indices
# =========================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LE_OUTER, LE_INNER, LE_TOP, LE_BOTTOM = 33, 133, 159, 145
RE_INNER, RE_OUTER, RE_TOP, RE_BOTTOM = 362, 263, 386, 374

UPPER_LIP = 13
LOWER_LIP = 14

NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


@dataclass
class TrackerConfig:
    ear_blink_threshold: float = 0.20
    min_blink_frames: int = 2
    gaze_dx_thresh: float = 0.11
    gaze_dy_thresh: float = 0.12
    yaw_thresh: float = 18.0
    yaw_soft: float = 10.0
    calib_seconds: float = 3.0
    smooth_window: int = 9
    min_hold_frames: int = 5

    # DRAW (Mode A full mesh by default)
    draw_face_mesh: bool = True
    draw_eye_dots: bool = True
    draw_iris_dots: bool = True
    draw_eye_boxes: bool = True

    write_annotated_video: bool = True

    downscale_width: Optional[int] = None
    downscale_height: Optional[int] = None
    mirror_lr: bool = False


def compute_gaze_xy_norm(lm_xy) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    left_iris = np.array([lm_xy(i) for i in LEFT_IRIS], dtype=np.float32)
    right_iris = np.array([lm_xy(i) for i in RIGHT_IRIS], dtype=np.float32)
    lc = left_iris.mean(axis=0)
    rc = right_iris.mean(axis=0)

    le_outer = np.array(lm_xy(LE_OUTER), dtype=np.float32)
    le_inner = np.array(lm_xy(LE_INNER), dtype=np.float32)
    le_top = np.array(lm_xy(LE_TOP), dtype=np.float32)
    le_bottom = np.array(lm_xy(LE_BOTTOM), dtype=np.float32)

    re_inner = np.array(lm_xy(RE_INNER), dtype=np.float32)
    re_outer = np.array(lm_xy(RE_OUTER), dtype=np.float32)
    re_top = np.array(lm_xy(RE_TOP), dtype=np.float32)
    re_bottom = np.array(lm_xy(RE_BOTTOM), dtype=np.float32)

    lx = clip01((lc[0] - le_outer[0]) / (le_inner[0] - le_outer[0] + 1e-6))
    rx = clip01((rc[0] - re_inner[0]) / (re_outer[0] - re_inner[0] + 1e-6))

    ly = clip01((lc[1] - le_top[1]) / (le_bottom[1] - le_top[1] + 1e-6))
    ry = clip01((rc[1] - re_top[1]) / (re_bottom[1] - re_top[1] + 1e-6))

    return float((lx + rx) / 2.0), float((ly + ry) / 2.0), lc, rc


def estimate_head_pose_yaw(lm_xy, w: int, h: int) -> Optional[float]:
    image_points = np.array(
        [lm_xy(NOSE_TIP), lm_xy(CHIN), lm_xy(LEFT_EYE_OUTER), lm_xy(RIGHT_EYE_OUTER), lm_xy(MOUTH_LEFT), lm_xy(MOUTH_RIGHT)],
        dtype=np.float32,
    )

    model_points = np.array(
        [(0.0, 0.0, 0.0),
         (0.0, -63.6, -12.5),
         (-43.3, 32.7, -26.0),
         (43.3, 32.7, -26.0),
         (-28.9, -28.9, -24.1),
         (28.9, -28.9, -24.1)],
        dtype=np.float32,
    )

    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotationMatrixToEulerAngles(R)
    return yaw


def classify_fused(dx: Optional[float], dy: Optional[float], yaw: Optional[float], cfg: TrackerConfig) -> str:
    if yaw is not None:
        if yaw >= cfg.yaw_thresh:
            return "RIGHT" if not cfg.mirror_lr else "LEFT"
        if yaw <= -cfg.yaw_thresh:
            return "LEFT" if not cfg.mirror_lr else "RIGHT"

    if dx is None or dy is None:
        return "UNKNOWN"

    ax, ay = abs(dx), abs(dy)

    if ax < cfg.gaze_dx_thresh and ay < cfg.gaze_dy_thresh:
        return "CENTER"

    if yaw is not None and abs(yaw) >= cfg.yaw_soft and ax < (cfg.gaze_dx_thresh * 1.3):
        if yaw > 0:
            return "RIGHT" if not cfg.mirror_lr else "LEFT"
        else:
            return "LEFT" if not cfg.mirror_lr else "RIGHT"

    if ax >= ay:
        return ("RIGHT" if dx > 0 else "LEFT") if not cfg.mirror_lr else ("LEFT" if dx > 0 else "RIGHT")
    else:
        return "DOWN" if dy > 0 else "UP"


def majority_vote(labels: deque) -> str:
    if not labels:
        return "UNKNOWN"
    return Counter(labels).most_common(1)[0][0]


def apply_hysteresis(current_label: str, proposed_label: str, hold_state: Dict[str, Any], cfg: TrackerConfig) -> str:
    if proposed_label == current_label:
        hold_state["candidate"] = proposed_label
        hold_state["count"] = 0
        return current_label

    if hold_state["candidate"] != proposed_label:
        hold_state["candidate"] = proposed_label
        hold_state["count"] = 1
        return current_label

    hold_state["count"] += 1
    if hold_state["count"] >= cfg.min_hold_frames:
        hold_state["candidate"] = proposed_label
        hold_state["count"] = 0
        return proposed_label

    return current_label


# =========================
# InsightFace matching (ROI-first)
# =========================
def init_face_app() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def get_ref_embedding(face_app: FaceAnalysis, ref_img_path: Path) -> np.ndarray:
    img = read_image_bgr(ref_img_path)
    faces = face_app.get(img)
    if not faces:
        raise RuntimeError(f"No face detected in reference image: {ref_img_path}")
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    emb = faces[0].embedding
    return emb / (np.linalg.norm(emb) + 1e-9)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


def expand_bbox(b: Tuple[int, int, int, int], w: int, h: int, pad: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(bw * pad)
    py = int(bh * pad)
    nx1 = max(0, x1 - px)
    ny1 = max(0, y1 - py)
    nx2 = min(w - 1, x2 + px)
    ny2 = min(h - 1, y2 + py)
    if nx2 <= nx1 + 2 or ny2 <= ny1 + 2:
        return b
    return (nx1, ny1, nx2, ny2)


def pick_candidate_face_bbox_roi(
    face_app: FaceAnalysis,
    frame_bgr: np.ndarray,
    ref_emb: Optional[np.ndarray],
    *,
    min_sim: float,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[float]]:
    if ref_emb is None:
        return None, None

    xoff = yoff = 0
    img = frame_bgr
    if roi is not None:
        x1, y1, x2, y2 = roi
        img = frame_bgr[y1:y2, x1:x2]
        xoff, yoff = x1, y1

    faces = face_app.get(img)
    if not faces:
        return None, None

    best = None
    best_sim = -1.0
    for f in faces:
        emb = f.embedding
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        s = cosine_sim(emb, ref_emb)
        if s > best_sim:
            best_sim = s
            best = f

    if best is None:
        return None, None

    x1, y1, x2, y2 = best.bbox.astype(int).tolist()
    bbox = (x1 + xoff, y1 + yoff, x2 + xoff, y2 + yoff)

    if best_sim < min_sim:
        return None, best_sim

    return bbox, best_sim


# =========================
# FaceMesh selection helpers
# =========================
def facemesh_bbox(face_lms, w: int, h: int) -> Tuple[float, float, float, float, float, float, float]:
    xs = [lm.x * w for lm in face_lms.landmark]
    ys = [lm.y * h for lm in face_lms.landmark]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return x1, y1, x2, y2, cx, cy, area


def pick_facemesh_face(
    res_multi_face_landmarks,
    w: int,
    h: int,
    prefer_bbox: Optional[Tuple[int, int, int, int]],
    *,
    candidate_side: str,
    side_margin_pct: float,
):
    faces = res_multi_face_landmarks or []
    if not faces:
        return None

    if prefer_bbox is not None:
        bx1, by1, bx2, by2 = prefer_bbox
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0
        best = None
        best_score = 1e18
        for f in faces:
            x1, y1, x2, y2, cx, cy, area = facemesh_bbox(f, w, h)
            d = (cx - bcx) ** 2 + (cy - bcy) ** 2
            if d < best_score:
                best_score = d
                best = f
        return best

    # fallback (no ref image): pick biggest face
    scored = []
    for f in faces:
        x1, y1, x2, y2, cx, cy, area = facemesh_bbox(f, w, h)
        scored.append((area, f))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1] if scored else None


# =========================
# Metrics + OpenAI rating
# =========================
def compute_blinks(df: pd.DataFrame, cfg: TrackerConfig) -> int:
    ear = pd.to_numeric(df["ear"], errors="coerce").to_numpy(dtype=np.float32)
    has_face = df["has_face"].to_numpy(dtype=np.int32)

    closed_run = 0
    blinks = 0
    was_closed = False

    for i in range(len(df)):
        if not has_face[i] or np.isnan(ear[i]):
            was_closed = False
            closed_run = 0
            continue

        is_closed = ear[i] < cfg.ear_blink_threshold
        if is_closed:
            closed_run += 1
            if closed_run >= cfg.min_blink_frames:
                was_closed = True
        else:
            if was_closed:
                blinks += 1
            was_closed = False
            closed_run = 0

    return blinks


def summarize_metrics(df: pd.DataFrame, fps: float, cfg: TrackerConfig) -> Dict[str, Any]:
    duration_sec = float(len(df) / fps) if len(df) else 0.0
    duration_min = max(duration_sec / 60.0, 1e-6)
    face_detected_pct = float(100.0 * df["has_face"].mean()) if len(df) else 0.0

    blinks = compute_blinks(df, cfg)
    blink_rate = float(blinks / duration_min)

    valid = df[(df["has_face"] == 1) & df["gaze_dir"].notna() & (df["gaze_dir"] != "UNKNOWN")]
    if len(valid) > 0:
        focus_on_camera_pct = float(100.0 * (valid["gaze_dir"] == "CENTER").mean())
        look_left_pct = float(100.0 * (valid["gaze_dir"] == "LEFT").mean())
        look_right_pct = float(100.0 * (valid["gaze_dir"] == "RIGHT").mean())
        look_up_pct = float(100.0 * (valid["gaze_dir"] == "UP").mean())
        look_down_pct = float(100.0 * (valid["gaze_dir"] == "DOWN").mean())
    else:
        focus_on_camera_pct = 0.0
        look_left_pct = look_right_pct = look_up_pct = look_down_pct = 0.0

    yaw_vals = (
        pd.to_numeric(df.loc[(df["has_face"] == 1), "yaw_deg"], errors="coerce")
        .dropna()
        .to_numpy(dtype=np.float32)
    )
    yaw_abs_mean = float(np.mean(np.abs(yaw_vals))) if len(yaw_vals) else None

    return {
        "duration_sec": round(duration_sec, 2),
        "face_detected_pct": round(face_detected_pct, 2),
        "blink_count": int(blinks),
        "blink_rate_per_min": round(blink_rate, 2),
        "focus_on_camera_pct": round(focus_on_camera_pct, 2),
        "look_left_pct": round(look_left_pct, 2),
        "look_right_pct": round(look_right_pct, 2),
        "look_up_pct": round(look_up_pct, 2),
        "look_down_pct": round(look_down_pct, 2),
        "yaw_abs_mean_deg": None if yaw_abs_mean is None else round(yaw_abs_mean, 3),
        "notes": "OK",
    }


def parse_json_strict(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return {"score": 0, "verdict": "FAIL", "reasoning": f"Could not parse JSON. Raw: {text[:500]}"}


def rate_with_openai(summary_payload: Dict[str, Any], model: str = DEFAULT_OPENAI_MODEL) -> Dict[str, Any]:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env / env vars.")
    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You rate interview camera focus using ONLY the metrics provided.\n"
        "Main signal: focus_on_camera_pct.\n"
        ">=80 excellent, 65-79 good, 50-64 weak, <50 poor.\n"
        "Penalize if face_detected_pct < 90.\n"
        "Return ONLY valid JSON with keys: score (0-100), verdict (PASS/FAIL), reasoning.\n"
        "Verdict PASS if score >= 70 else FAIL.\n"
        "Reasoning must mention the key metrics with numbers."
    )

    user_content = "Metrics summary:\n" + json.dumps(summary_payload, indent=2)

    schema = {
        "name": "video_behavior_score",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 100},
                "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
                "reasoning": {"type": "string"},
            },
            "required": ["score", "verdict", "reasoning"],
        },
    }

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            text={"format": {"type": "json_schema", "json_schema": schema}},
            temperature=0,
        )
        return parse_json_strict(resp.output_text)
    except Exception:
        resp2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        text = resp2.choices[0].message.content or "{}"
        return parse_json_strict(text)


# =========================
# ffmpeg helpers
# =========================
def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg and restart terminal.")


def transcode_to_h264_faststart(src_mp4: Path, dst_mp4: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src_mp4),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst_mp4),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode failed:\n{p.stderr[:1500]}")


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "file"


def is_video_file(name: str) -> bool:
    p = Path(name)
    n = p.name.lower()
    if p.suffix.lower() not in VIDEO_EXTS:
        return False
    if n.startswith("."):
        return False
    if "__eye_" in n:
        return False
    return True


# =========================
# Candidate image discovery
# =========================
def find_candidate_image_in_person_folder(service, person_folder_id: str) -> Optional[dict]:
    files = list(drive_list_children(service, person_folder_id, None))
    imgs = []
    for f in files:
        if f.get("mimeType") == FOLDER_MIME:
            continue
        name = f.get("name") or ""
        if CANDIDATE_IMAGE_NAME_RX.search(name):
            imgs.append(f)
    if not imgs:
        return None
    imgs.sort(key=lambda x: (x.get("modifiedTime") or ""), reverse=True)
    return imgs[0]


# =========================
# MAIN tracking function
# =========================
def track_one_video(
    video_path: Path,
    out_csv: Path,
    out_annot_tmp_mp4: Path,
    out_annot_h264_mp4: Path,
    cfg: TrackerConfig,
    face_app: FaceAnalysis,
    ref_emb: Optional[np.ndarray],
    *,
    candidate_side: str,
    side_margin_pct: float,
    match_every_n: int,
    ref_min_sim: float,
    openai_model: str = DEFAULT_OPENAI_MODEL,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if STRICT_REF_ONLY and ref_emb is not None and match_every_n == 0:
        raise RuntimeError("STRICT_REF_ONLY requires match_every_n > 0 (ref matching cannot be disabled).")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0

    rows: List[Dict[str, Any]] = []
    calib_x: List[float] = []
    calib_y: List[float] = []
    calib_frames_needed = int(max(cfg.calib_seconds * fps, 1))

    valid_gaze_count = 0
    stable_center_count = 0

    hist = deque(maxlen=max(cfg.smooth_window, 1))
    stable_label = "UNKNOWN"
    hold_state = {"candidate": "UNKNOWN", "count": 0}

    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    writer = None

    last_good_match_frame: Optional[int] = None
    MAX_STALE_FRAMES = max(1, int(match_every_n * max(0.5, MAX_STALE_MULT)))
    match_count = 0

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as fm:
        center_x = None
        center_y = None
        frame_idx = 0

        cached_bbox: Optional[Tuple[int, int, int, int]] = None
        cached_sim: Optional[float] = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if cfg.downscale_width and cfg.downscale_height:
                frame = cv2.resize(frame, (int(cfg.downscale_width), int(cfg.downscale_height)), interpolation=cv2.INTER_AREA)

            h, w = frame.shape[:2]

            if writer is None and cfg.write_annotated_video:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_annot_tmp_mp4), fourcc, float(fps), (w, h))

            t = frame_idx / fps

            # ---------- Ref match (ROI-first) ----------
            if ref_emb is not None and match_every_n > 0 and (frame_idx % match_every_n == 0):
                match_count += 1

                roi = None
                if cached_bbox is not None:
                    roi = expand_bbox(cached_bbox, w, h, pad=ROI_PAD)

                # ROI attempt
                bbox, sim = pick_candidate_face_bbox_roi(face_app, frame, ref_emb, min_sim=ref_min_sim, roi=roi)

                # periodic full-frame reacquire
                if bbox is None and (match_count % max(1, FULL_FRAME_EVERY_K_MATCHES) == 0):
                    bbox, sim = pick_candidate_face_bbox_roi(face_app, frame, ref_emb, min_sim=ref_min_sim, roi=None)

                cached_bbox = bbox
                cached_sim = sim
                if bbox is not None:
                    last_good_match_frame = frame_idx

            # Expire stale match
            if ref_emb is not None and last_good_match_frame is not None:
                if (frame_idx - last_good_match_frame) > MAX_STALE_FRAMES:
                    cached_bbox = None
                    cached_sim = None
                    last_good_match_frame = None

            # STRICT trainer frame check
            force_no_face_this_frame = False
            if STRICT_REF_ONLY and ref_emb is not None:
                if (cached_bbox is None) or (cached_sim is None) or (cached_sim < ref_min_sim):
                    force_no_face_this_frame = True

            # Defaults
            has_face = False
            ear = None
            gaze_x = gaze_y = None
            mouth_open = None
            yaw_deg = None
            raw = "UNKNOWN"
            lc = rc = None
            le_box = re_box = None

            # Do NOT run FaceMesh for trainer frames
            res = None
            if not force_no_face_this_frame:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)

            if res and res.multi_face_landmarks:
                face_lms = pick_facemesh_face(
                    res.multi_face_landmarks, w, h, cached_bbox,
                    candidate_side=candidate_side, side_margin_pct=side_margin_pct
                )
                if face_lms is not None:
                    has_face = True

                    # Draw mesh (Mode A)
                    if cfg.draw_face_mesh and writer is not None:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_lms,
                            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                        )
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_lms,
                            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
                        )

                    lm = face_lms.landmark

                    def lm_xy(i: int) -> Tuple[float, float]:
                        return (lm[i].x * w, lm[i].y * h)

                    left_eye_pts = [lm_xy(i) for i in LEFT_EYE]
                    right_eye_pts = [lm_xy(i) for i in RIGHT_EYE]
                    ear = (eye_aspect_ratio(left_eye_pts) + eye_aspect_ratio(right_eye_pts)) / 2.0

                    gaze_x, gaze_y, lc, rc = compute_gaze_xy_norm(lm_xy)
                    yaw_deg = estimate_head_pose_yaw(lm_xy, w, h)

                    # Eye boxes
                    le_outer = np.array(lm_xy(LE_OUTER), dtype=np.float32)
                    le_inner = np.array(lm_xy(LE_INNER), dtype=np.float32)
                    le_top = np.array(lm_xy(LE_TOP), dtype=np.float32)
                    le_bottom = np.array(lm_xy(LE_BOTTOM), dtype=np.float32)
                    le_box = (le_outer, le_inner, le_top, le_bottom)

                    re_inner = np.array(lm_xy(RE_INNER), dtype=np.float32)
                    re_outer = np.array(lm_xy(RE_OUTER), dtype=np.float32)
                    re_top = np.array(lm_xy(RE_TOP), dtype=np.float32)
                    re_bottom = np.array(lm_xy(RE_BOTTOM), dtype=np.float32)
                    re_box = (re_inner, re_outer, re_top, re_bottom)

                    if frame_idx < calib_frames_needed and gaze_x is not None and gaze_y is not None:
                        calib_x.append(gaze_x)
                        calib_y.append(gaze_y)

                    if center_x is None and frame_idx >= calib_frames_needed:
                        center_x = robust_center(calib_x)
                        center_y = robust_center(calib_y)
                    if center_x is None:
                        center_x = 0.5
                        center_y = 0.5

                    gaze_dx = gaze_dy = None
                    if gaze_x is not None and gaze_y is not None and not (np.isnan(gaze_x) or np.isnan(gaze_y)):
                        gaze_dx = float(gaze_x - center_x)
                        gaze_dy = float(gaze_y - center_y)
                        raw = classify_fused(gaze_dx, gaze_dy, yaw_deg, cfg)

                    # Draw eye dots/iris/boxes
                    if writer is not None:
                        if cfg.draw_eye_dots:
                            for p in left_eye_pts + right_eye_pts:
                                cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
                        if cfg.draw_iris_dots and lc is not None and rc is not None:
                            cv2.circle(frame, (int(lc[0]), int(lc[1])), 4, (0, 0, 255), -1)
                            cv2.circle(frame, (int(rc[0]), int(rc[1])), 4, (0, 0, 255), -1)
                        if cfg.draw_eye_boxes and le_box is not None and re_box is not None:
                            le_outer, le_inner, le_top, le_bottom = le_box
                            re_inner, re_outer, re_top, re_bottom = re_box
                            le_min = (int(min(le_outer[0], le_inner[0])), int(min(le_top[1], le_bottom[1])))
                            le_max = (int(max(le_outer[0], le_inner[0])), int(max(le_top[1], le_bottom[1])))
                            re_min = (int(min(re_inner[0], re_outer[0])), int(min(re_top[1], re_bottom[1])))
                            re_max = (int(max(re_inner[0], re_outer[0])), int(max(re_top[1], re_bottom[1])))
                            cv2.rectangle(frame, le_min, le_max, (255, 255, 0), 1)
                            cv2.rectangle(frame, re_min, re_max, (255, 255, 0), 1)

                    # Mouth open
                    upper = np.array(lm_xy(UPPER_LIP), dtype=np.float32)
                    lower = np.array(lm_xy(LOWER_LIP), dtype=np.float32)
                    mouth_gap = float(np.linalg.norm(upper - lower))
                    eye_span = float(np.linalg.norm(np.array(lm_xy(LEFT_EYE_OUTER), dtype=np.float32) - np.array(lm_xy(RIGHT_EYE_OUTER), dtype=np.float32)))
                    mouth_open = float(mouth_gap / (eye_span + 1e-6))

            # smoothing only if not trainer frame
            if not force_no_face_this_frame:
                hist.append(raw)
                proposed = majority_vote(hist)
                if stable_label == "UNKNOWN":
                    stable_label = proposed
                else:
                    stable_label = apply_hysteresis(stable_label, proposed, hold_state, cfg)

                if has_face and stable_label != "UNKNOWN":
                    valid_gaze_count += 1
                    if stable_label == "CENTER":
                        stable_center_count += 1

            # draw labels
            if writer is not None:
                if force_no_face_this_frame and (ref_emb is not None):
                    put_text_with_bg(frame, "TRAINER FRAME (ref mismatch)", (10, 40), font_scale=1.0, thickness=2)
                else:
                    focus_pct = (100.0 * stable_center_count / max(valid_gaze_count, 1))
                    label = f"LOOKING: {stable_label}   Focus:{focus_pct:.1f}%"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    x = max(int((w - tw) / 2), 10)
                    put_text_with_bg(frame, label, (x, 40), font_scale=1.0, thickness=2)

                if ref_emb is not None:
                    if cached_sim is None:
                        put_text_with_bg(frame, "REF_SIM: -", (10, h - 20), font_scale=0.8, thickness=2)
                    else:
                        put_text_with_bg(frame, f"REF_SIM: {cached_sim:.3f}", (10, h - 20), font_scale=0.8, thickness=2)

                writer.write(frame)

            # ✅ trainer frames should not inherit stable label in CSV
            row_raw = raw
            row_stable = stable_label
            if force_no_face_this_frame and (ref_emb is not None):
                row_raw = "UNKNOWN"
                row_stable = "UNKNOWN"

            accepted_match = (ref_emb is not None) and (cached_sim is not None) and (cached_sim >= ref_min_sim) and (cached_bbox is not None)

            rows.append({
                "frame": frame_idx,
                "time_sec": t,
                "has_face": int(has_face),
                "ear": ear,
                "gaze_x_norm": gaze_x,
                "gaze_y_norm": gaze_y,
                "yaw_deg": yaw_deg,
                "gaze_dir_raw": row_raw,
                "gaze_dir": row_stable,
                "mouth_open_norm": mouth_open,
                "ref_sim": None if cached_sim is None else float(cached_sim),
                "ref_matched": int(accepted_match),
            })

            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)  # local-only

    if cfg.write_annotated_video:
        if not out_annot_tmp_mp4.exists():
            raise RuntimeError("Annotated temp video was not created (unexpected).")
        transcode_to_h264_faststart(out_annot_tmp_mp4, out_annot_h264_mp4)

    meta = {
        "video_path": str(video_path),
        "fps": float(fps),
        "total_frames_reported": int(len(df)),
        "total_frames_metadata": int(total_frames_meta),
        "duration_sec": float((len(df) / fps) if len(df) else 0.0),
        "ref_min_sim": float(ref_min_sim),
        "match_every_n": int(match_every_n),
        "strict_ref_only": bool(STRICT_REF_ONLY),
        "has_reference_image": bool(ref_emb is not None),
        "roi_pad": float(ROI_PAD),
        "full_frame_every_k_matches": int(FULL_FRAME_EVERY_K_MATCHES),
        "max_stale_frames": int(MAX_STALE_FRAMES),
        "downscale": f"{cfg.downscale_width}x{cfg.downscale_height}" if (cfg.downscale_width and cfg.downscale_height) else "",
    }

    summary = summarize_metrics(df, float(fps), cfg)
    summary_full = {"video_meta": meta, "summary": summary}
    result = rate_with_openai(summary_full, model=openai_model)
    return summary_full, result


# =========================
# CLI (kept minimal; defaults already FAST)
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="FAST default: python test6.py")
    p.add_argument("--candidate_side", choices=["auto", "left", "right"], default="auto")
    p.add_argument("--side_margin_pct", type=float, default=0.06)
    p.add_argument("--ref_min_sim", type=float, default=0.30)
    p.add_argument("--match_every_n", type=int, default=FAST_DEFAULT_MATCH_EVERY_N)
    p.add_argument("--downscale", type=str, default=FAST_DEFAULT_DOWNSCALE,
                   help="Processing resolution. Annotated output uses this resolution. Default: 1280x720 (can set env FAST_DOWNSCALE).")
    p.add_argument("--mirror_lr", action="store_true")
    return p.parse_args()


def main():
    ensure_ffmpeg()

    # Use all CPU threads in OpenCV
    try:
        cv2.setNumThreads(_CPU)
    except Exception:
        pass

    args = parse_args()
    service = get_drive_service()

    candidates_root = drive_search_folder_anywhere(service, ROOT_FOLDER_NAME)
    if not candidates_root:
        raise RuntimeError(f"Could not find folder '{ROOT_FOLDER_NAME}' anywhere in Drive.")
    base_root = pick_best_named_folder(candidates_root)
    slots_parent_id = base_root["id"]

    slot = choose_slot(service, slots_parent_id)
    slot_name = slot["name"]
    slot_id = slot["id"]

    face_app = init_face_app()

    # Parse downscale default
    down_w = down_h = None
    if args.downscale.strip():
        m = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", args.downscale.strip())
        if not m:
            raise RuntimeError("--downscale must be like 1280x720")
        down_w, down_h = int(m.group(1)), int(m.group(2))

    cfg = TrackerConfig(
        # thresholds
        ear_blink_threshold=0.20,
        min_blink_frames=2,
        gaze_dx_thresh=0.11,
        gaze_dy_thresh=0.12,
        yaw_thresh=18.0,
        yaw_soft=10.0,
        calib_seconds=3.0,
        smooth_window=9,
        min_hold_frames=5,

        # ✅ Mode A: FULL MESH + dots/boxes (like screenshot)
        draw_face_mesh=True,
        draw_eye_dots=True,
        draw_iris_dots=True,
        draw_eye_boxes=True,

        # ✅ annotated ON always
        write_annotated_video=True,

        downscale_width=down_w,
        downscale_height=down_h,
        mirror_lr=bool(args.mirror_lr),
    )

    people = sorted(
        [f for f in drive_list_children(service, slot_id, FOLDER_MIME) if (f.get("name") or "").strip() not in SKIP_PERSON_FOLDERS],
        key=lambda x: (x.get("name") or "").lower(),
    )

    print("\n" + "=" * 100)
    print(f"RUN ROOT: {base_root['name']}   SLOT: {slot_name}")
    print(f"People: {len(people)}")
    print(f"FAST DEFAULTS -> match_every_n={args.match_every_n}  downscale='{args.downscale}'  annotated=ON  mesh=ON")
    print(f"STRICT_REF_ONLY={STRICT_REF_ONLY} (if ref exists, ignore frames unless REF_SIM >= ref_min_sim)")
    print("UPLOADS: result.json + annotated video. NOT uploading summary.json or metrics.csv.")
    print(f"ROI: ROI_PAD={ROI_PAD} FULL_FRAME_EVERY_K_MATCHES={FULL_FRAME_EVERY_K_MATCHES} MAX_STALE_MULT={MAX_STALE_MULT}")
    print(f"VIDEO RETRIES: MAX_VIDEO_RETRIES={MAX_VIDEO_RETRIES} base_sleep={VIDEO_RETRY_BASE_SLEEP}s")
    if SLOT_CHOICE:
        print(f"AUTO SLOT_CHOICE={SLOT_CHOICE}")
    if HEADLESS_AUTH:
        print("HEADLESS_AUTH=1 (console OAuth)")
    print("=" * 100)

    for person in people:
        person_name = person["name"]
        person_id = person["id"]

        print("\n" + "-" * 100)
        print(f"PERSON: {person_name}")

        ref_file = find_candidate_image_in_person_folder(service, person_id)
        ref_emb = None

        with tempfile.TemporaryDirectory() as td_all:
            td_all = Path(td_all)

            if ref_file:
                try:
                    ext = Path(ref_file["name"]).suffix.lower()
                    local_ref = td_all / f"__candidate_ref__{sanitize_filename(Path(ref_file['name']).stem)}{ext}"
                    print(f"  [REF ] Downloading candidate image: {ref_file['name']}")
                    drive_download_file(service, ref_file["id"], local_ref)
                    ref_emb = get_ref_embedding(face_app, local_ref)
                    print("  [REF ] Face reference embedding ready.")
                except Exception as e:
                    print(f"  [WARN] Candidate image exists but failed to use it: {e}")
                    ref_emb = None
            else:
                print("  [WARN] No candidate image found. Fallback will use largest-face logic.")

            for deliverable_name in FOLDER_NAMES_TO_PROCESS:
                deliverable_folder = drive_find_child(service, person_id, deliverable_name, FOLDER_MIME)
                if not deliverable_folder:
                    continue

                deliverable_id = deliverable_folder["id"]
                files = list(drive_list_children(service, deliverable_id, None))

                videos = [f for f in files if f.get("mimeType") != FOLDER_MIME and is_video_file(f.get("name") or "")]
                videos.sort(key=lambda x: (x.get("name") or "").lower())
                if not videos:
                    continue

                print(f"\n  FOLDER: {deliverable_name}   Videos: {len(videos)}")

                for v in videos:
                    vname = v["name"]
                    safe_vname = sanitize_filename(vname)
                    stem = Path(safe_vname).stem

                    out_annot_name = f"{stem}{ANNOT_SUFFIX}"
                    out_summary_name = f"{stem}{SUMMARY_SUFFIX}"   # local-only
                    out_result_name = f"{stem}{RESULT_SUFFIX}"
                    out_metrics_name = f"{stem}{METRICS_SUFFIX}"   # local-only

                    # Skip if already processed (result is marker)
                    existing_result = drive_find_child(service, deliverable_id, out_result_name, None)
                    if existing_result:
                        print(f"    [SKIP] {vname} -> already has {out_result_name}")
                        continue

                    print(f"    [RUN ] {vname}")

                    with tempfile.TemporaryDirectory() as td:
                        td = Path(td)
                        local_video = td / safe_vname
                        local_csv = td / out_metrics_name
                        local_tmp_annot = td / (stem + "__tmp.mp4")
                        local_annot_h264 = td / out_annot_name
                        local_summary = td / out_summary_name
                        local_result = td / out_result_name

                        attempt = 0
                        while attempt < MAX_VIDEO_RETRIES:
                            attempt += 1
                            try:
                                # clean remnants
                                for p in [local_video, local_csv, local_tmp_annot, local_annot_h264, local_summary, local_result]:
                                    try:
                                        if p.exists():
                                            p.unlink()
                                    except Exception:
                                        pass

                                print("      [DL  ] downloading video")
                                drive_download_file(service, v["id"], local_video)

                                summary_full, result = track_one_video(
                                    video_path=local_video,
                                    out_csv=local_csv,
                                    out_annot_tmp_mp4=local_tmp_annot,
                                    out_annot_h264_mp4=local_annot_h264,
                                    cfg=cfg,
                                    face_app=face_app,
                                    ref_emb=ref_emb,
                                    candidate_side=args.candidate_side,
                                    side_margin_pct=args.side_margin_pct,
                                    match_every_n=args.match_every_n,
                                    ref_min_sim=args.ref_min_sim,
                                    openai_model=DEFAULT_OPENAI_MODEL,
                                )

                                # Write locally (DO NOT upload summary/metrics)
                                local_summary.write_text(json.dumps(summary_full, indent=2), encoding="utf-8")
                                local_result.write_text(json.dumps(result, indent=2), encoding="utf-8")

                                print("      [UP  ] uploading annotated video (H.264)")
                                drive_upload_file(service, deliverable_id, out_annot_name, local_annot_h264, "video/mp4")

                                print("      [UP  ] uploading result.json")
                                drive_upload_file(service, deliverable_id, out_result_name, local_result, "application/json")

                                print("      [OK  ] done")
                                break

                            except Exception as e:
                                retryable = is_retryable_video_error(e)
                                print(f"      [FAIL] {vname} (attempt {attempt}/{MAX_VIDEO_RETRIES}) -> {type(e).__name__}: {e}")

                                if (attempt >= MAX_VIDEO_RETRIES) or (not retryable):
                                    if not retryable:
                                        print("      [STOP] Non-retryable error. Moving to next video.")
                                    else:
                                        print("      [GIVE] Max retries reached. Moving to next video.")
                                    break

                                sleep_s = (VIDEO_RETRY_BASE_SLEEP * (2 ** (attempt - 1))) + random.random()
                                print(f"      [RETRY] waiting {sleep_s:.1f}s then retrying same video...")
                                time.sleep(sleep_s)

                        time.sleep(0.25)

    print("\nDONE.")


if __name__ == "__main__":
    main()

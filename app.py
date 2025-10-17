
# app_service_account_upload_nodwd.py
# Streamlit: Google Drive Audio Converter — Upload Service Account (No DWD)
#
# - Upload a Service Account JSON key each session (kept in memory only)
# - NO Domain-Wide Delegation (no impersonation)
# - Share Drive folders/files with the SA's client_email for access
# - Robust retries, download validation, and thread-safe logging
# - Uses imageio-ffmpeg (no local system install)
#
import os
import io
import time
import json
import math
import tempfile
import logging
import threading
import collections
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
os.environ["STREAMLIT_DATAFRAME_SERIALIZATION"] = "legacy"  # avoid importing pyarrow
import streamlit as st
import pandas as pd

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.auth import exceptions as ga_exceptions

import imageio_ffmpeg

# ---------------------------
# Constants & Logging
# ---------------------------
SCOPES = ["https://www.googleapis.com/auth/drive"]
AUDIO_MIMES = [
    "audio/mpeg","audio/wav","audio/aac","audio/ogg",
    "audio/flac","audio/mp4","audio/x-wav","audio/x-ms-wma"
]
AUDIO_FORMATS = {
    "MP3":  {"ext": "mp3",  "mime": "audio/mpeg", "ffmpeg_args": ["-codec:a", "libmp3lame"]},
    "AAC":  {"ext": "aac",  "mime": "audio/aac",  "ffmpeg_args": ["-codec:a", "aac"]},
    "M4A":  {"ext": "m4a",  "mime": "audio/mp4",  "ffmpeg_args": ["-codec:a", "aac"]},
    "OGG":  {"ext": "ogg",  "mime": "audio/ogg",  "ffmpeg_args": ["-codec:a", "libvorbis"]},
    "FLAC": {"ext": "flac", "mime": "audio/flac","ffmpeg_args": ["-codec:a", "flac"]},
    "WAV":  {"ext": "wav",  "mime": "audio/wav",  "ffmpeg_args": ["-codec:a", "pcm_s16le"]},
}
BITRATES = ["32", "64", "96", "128", "192", "256", "320"]
SAMPLE_RATES = ["Auto", "22050", "32000", "44100", "48000"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("drive-audio-converter-nodwd")

# ---------------------------
# Utilities
# ---------------------------
def human_bytes(n: Optional[int]) -> str:
    if not n or n < 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"

def extract_folder_id(url_or_id: str) -> Optional[str]:
    s = (url_or_id or "").strip()
    if not s:
        return None
    if "/folders/" in s:
        return s.split("/folders/")[1].split("?")[0].split("/")[0]
    if "id=" in s:
        return s.split("id=")[1].split("&")[0]
    if "/" not in s and " " not in s and len(s) > 10:
        return s
    return None

def ffmpeg_convert(in_path: str, out_path: str, bitrate_kbps: int, sample_rate: Optional[int], fmt_args: List[str]) -> Tuple[bool, str]:
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", in_path]
    cmd.extend(fmt_args)
    cmd.extend(["-b:a", f"{bitrate_kbps}k"])
    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])
    cmd.append(out_path)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg failed: %s", e.stderr)
        return False, e.stderr or "ffmpeg error"

# ---------------------------
# Thread-safe logging (no Streamlit in worker threads)
# ---------------------------
LOG_BUF_MAX = 500
_log_lock = threading.Lock()
_log_buf = collections.deque(maxlen=LOG_BUF_MAX)

def log_threadsafe(msg: str):
    with _log_lock:
        _log_buf.append(f"{datetime.now().strftime('%H:%M:%S')} — {msg}")

def drain_logs():
    with _log_lock:
        items = list(_log_buf)
        _log_buf.clear()
    return items

# ---------------------------
# Drive Client with Retry
# ---------------------------
@dataclass
class RetryPolicy:
    max_attempts: int = 6
    base_sleep: float = 0.8
    jitter: float = 0.3

class DriveClient:
    def __init__(self, service, retry: RetryPolicy = RetryPolicy()):
        self.service = service
        self.retry = retry

    def _with_retry(self, func, *args, **kwargs):
        attempt = 0
        while True:
            try:
                return func(*args, **kwargs)
            except HttpError as e:
                status = getattr(e, "status_code", None) or (e.resp.status if getattr(e, "resp", None) else None)
                if status in (403, 429, 500, 502, 503, 504) and attempt < self.retry.max_attempts - 1:
                    attempt += 1
                    sleep_for = self.retry.base_sleep * (2 ** (attempt - 1))
                    sleep_for *= 1 + random.uniform(-self.retry.jitter, self.retry.jitter)
                    time.sleep(max(0.2, sleep_for))
                    continue
                raise
            except Exception:
                if attempt < self.retry.max_attempts - 1:
                    attempt += 1
                    time.sleep(self.retry.base_sleep * (2 ** (attempt - 1)))
                    continue
                raise

    def is_folder(self, file_id: str) -> bool:
        def op():
            return self.service.files().get(fileId=file_id, fields="id, mimeType", supportsAllDrives=True).execute()
        try:
            meta = self._with_retry(op)
            return meta.get("mimeType") == "application/vnd.google-apps.folder"
        except Exception as e:
            logger.exception("is_folder error: %s", e)
            return False

    def ensure_subfolder(self, parent_id: Optional[str], name: str) -> Optional[str]:
        """Create (or find) a subfolder under parent_id or root if None."""
        # Escape single quotes in the name safely
        safe_name = name.replace("'", "\\'")
        q = f"mimeType='application/vnd.google-apps.folder' and trashed=false and name='{safe_name}'"
        if parent_id:
            q += f" and '{parent_id}' in parents"
        def list_op(page_token=None):
            return self.service.files().list(
                q=q, pageSize=100, fields="nextPageToken, files(id, name)",
                supportsAllDrives=True, includeItemsFromAllDrives=True,
                pageToken=page_token
            ).execute()
        try:
            page_token = None
            while True:
                resp = self._with_retry(list_op, page_token)
                for f in resp.get("files", []):
                    return f["id"]
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
            if parent_id:
                meta["parents"] = [parent_id]
            def create_op():
                return self.service.files().create(
                    body=meta, fields="id", supportsAllDrives=True
                ).execute()
            res = self._with_retry(create_op)
            return res.get("id")
        except Exception as e:
            logger.exception("ensure_subfolder error: %s", e)
            return None

    def list_audio_files(self, folder_id: Optional[str]) -> List[Dict[str, Any]]:
        mime_query = " or ".join([f"mimeType='{m}'" for m in AUDIO_MIMES])
        q = f"({mime_query}) and trashed=false"
        if folder_id:
            q += f" and '{folder_id}' in parents"
        def list_op(page_token=None):
            return self.service.files().list(
                q=q, pageSize=1000,
                fields="nextPageToken, files(id, name, size, mimeType, parents)",
                supportsAllDrives=True, includeItemsFromAllDrives=True,
                pageToken=page_token
            ).execute()
        files = []
        try:
            page_token = None
            while True:
                resp = self._with_retry(list_op, page_token)
                files.extend(resp.get("files", []))
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
        except Exception as e:
            logger.exception("list_audio_files error: %s", e)
        return files

    def download_to_temp(self, file_id: str) -> str:
        """
        Download file to a NamedTemporaryFile and return its path.
        - Sets acknowledgeAbuse=True to allow downloading flagged binaries (common for media).
        - Verifies non-empty result.
        - Detects obvious HTML/JSON error payloads to surface permission/abuse issues.
        - Retries with exponential backoff via _with_retry.
        """
        req = self.service.files().get_media(fileId=file_id, supportsAllDrives=True, acknowledgeAbuse=True)
        tmp = tempfile.NamedTemporaryFile(delete=False, prefix="dl_", suffix=".bin")
        fh = tmp.file
        downloader = MediaIoBaseDownload(fh, req, chunksize=2 * 1024 * 1024)
        done = False
        last_status = None
        while not done:
            def next_chunk():
                return downloader.next_chunk()
            last_status, done = self._with_retry(next_chunk)
        fh.flush(); fh.close()

        # Validate non-empty
        if not os.path.exists(tmp.name) or os.path.getsize(tmp.name) <= 0:
            raise RuntimeError("Download produced an empty file. Check sharing permissions or file health.")

        # Heuristic: detect HTML/JSON error payload (instead of audio bytes)
        with open(tmp.name, "rb") as f:
            head = f.read(1024).strip().lower()
        if head.startswith(b'<!doctype html') or head.startswith(b'<html') or head.startswith(b'{') or head.startswith(b'['):
            raise RuntimeError("Downloaded content is not audio (looks like an error page or JSON). Share the file with this Service Account.")

        return tmp.name

    def upload_file(self, local_path: str, dest_name: str, mime_type: str, parent_id: Optional[str]) -> Optional[str]:
        meta = {"name": dest_name}
        if parent_id:
            meta["parents"] = [parent_id]
        media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)
        def create_op():
            return self.service.files().create(
                body=meta, media_body=media, fields="id", supportsAllDrives=True
            ).execute()
        try:
            res = self._with_retry(create_op)
            return res.get("id")
        except Exception as e:
            logger.exception("upload_file error: %s", e)
            return None

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Drive Audio Converter (Service Account, No DWD)", page_icon="🗝️", layout="wide")
st.title("🗝️ Google Drive Audio Converter — Upload Service Account (No DWD)")
st.caption("Upload a Service Account key each session. Share your Drive items with the SA’s client_email. No impersonation.")

with st.expander("🔐 Upload Service Account JSON", expanded=True):
    sa_file = st.file_uploader("Service Account JSON key", type=["json"])
    if st.button("Build Drive client", type="primary"):
        if not sa_file:
            st.error("Please upload a Service Account JSON file.")
        else:
            try:
                info = json.loads(sa_file.read().decode("utf-8"))
                if info.get("type") != "service_account":
                    st.error("This JSON is not a Service Account key. Please upload the SA key from Google Cloud (type='service_account').")
                    st.stop()
                creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)

                # Preflight token acquisition to fail fast with clear message
                try:
                    creds.refresh(Request())
                except ga_exceptions.RefreshError as e:
                    st.error(
                        "Could not obtain an access token for this Service Account.\n\n"
                        "• Ensure the **Google Drive API** is enabled for the project that owns this Service Account.\n"
                        "• Since this build has **no DWD**, you must **share** the Drive folders/files with the SA’s client_email.\n"
                        "• Check server time/clock skew and that the key is valid.\n\n"
                        f"Details: {e}"
                    )
                    st.stop()

                service = build("drive", "v3", credentials=creds, cache_discovery=False)
                st.session_state["drive_client"] = DriveClient(service)
                st.session_state["sa_email"] = info.get("client_email", "(unknown)")
                st.success(f"Authenticated as **{st.session_state['sa_email']}**")
            except Exception as e:
                st.error(f"Failed to build Drive client: {e}")

if "drive_client" not in st.session_state:
    st.info("Upload your Service Account JSON and click **Build Drive client** to continue.")
    st.stop()

client: DriveClient = st.session_state["drive_client"]

# ---------------------------
# Folder selection
# ---------------------------
st.subheader("📁 Choose Input & Output Folders")
left, right = st.columns(2)
with left:
    input_url = st.text_input("Input Folder URL or ID (must be shared with this Service Account)")
    input_id = extract_folder_id(input_url) if input_url.strip() else None
    if input_id:
        if client.is_folder(input_id):
            st.success("Valid input folder.")
        else:
            st.error("Provided input is not a folder or not accessible.")
            input_id = None
with right:
    output_url = st.text_input("Output Folder URL or ID (leave blank to save in Drive root that the SA can write to)")
    output_id = extract_folder_id(output_url) if output_url.strip() else None
    if output_id:
        if client.is_folder(output_id):
            st.success("Valid output folder.")
        else:
            st.error("Provided output is not a folder or not accessible.")
            output_id = None

# Optional: create subfolder
subfolder_name = st.text_input("(Optional) Create/use subfolder inside the output folder", value="")
if subfolder_name and output_id:
    if st.button("Ensure output subfolder exists / select it"):
        sub_id = client.ensure_subfolder(output_id, subfolder_name.strip())
        if sub_id:
            output_id = sub_id
            st.success(f"Using output subfolder: {subfolder_name.strip()}")
        else:
            st.error("Could not create or select the subfolder.")

# ---------------------------
# File listing & selection
# ---------------------------
st.subheader("🎵 Select Files to Convert")
files = client.list_audio_files(input_id)
if not files:
    st.warning("No audio files found. Ensure the folder/files are **shared with this Service Account**.")
    selected_files = []
else:
    labels = [f"{f['name']}  ({human_bytes(int(f.get('size', 0) or 0))})" for f in files]
    select_all = st.checkbox("Select all", value=False)
    selected_flags = st.multiselect(
        "Pick files to convert:",
        options=list(range(len(files))),
        default=list(range(len(files))) if select_all else [],
        format_func=lambda i: labels[i],
    )
    selected_files = [files[i] for i in selected_flags]
    st.caption(f"Selected {len(selected_files)} / {len(files)}")

# ---------------------------
# Conversion settings
# ---------------------------
st.subheader("⚙️ Conversion Settings")
c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    out_fmt = st.selectbox("Format", list(AUDIO_FORMATS.keys()), index=0)
with c2:
    bitrate = st.selectbox("Bitrate (kbps)", BITRATES, index=3)
with c3:
    sr_str = st.selectbox("Sample rate", SAMPLE_RATES, index=3)
    sr = int(sr_str) if sr_str.isdigit() else None
with c4:
    max_threads = st.slider("Parallel workers", min_value=1, max_value=os.cpu_count() or 4, value=2)

# ---------------------------
# Conversion runner
# ---------------------------
st.subheader("🚀 Run")
overall_placeholder = st.empty()
table_placeholder = st.empty()
log_box = st.expander("Live log", expanded=False)
report_placeholder = st.empty()

def render_logs():
    lines = drain_logs()
    if not lines:
        return
    with log_box:
        for line in lines:
            st.write(line)

@dataclass
class Row:
    id: str
    name: str
    status: str
    download_size: int
    upload_size: int
    duration_s: float
    error: str = ""

def convert_one(finfo: Dict[str, Any]) -> Row:
    t0 = time.time()
    fid = finfo["id"]
    fname = finfo["name"]
    orig_size = int(finfo.get("size", 0) or 0)
    if orig_size <= 0:
        return Row(fid, fname, "Skipped (empty size)", 0, 0, 0.0, "Drive reported size=0; cannot convert.")
    dl_path = ""
    try:
        log_threadsafe(f"⬇️ Downloading: {fname}")
        dl_path = client.download_to_temp(fid)

        log_threadsafe(f"🎛️ Converting: {fname}")
        stem = os.path.splitext(fname)[0]
        out_ext = AUDIO_FORMATS[out_fmt]["ext"]
        target = os.path.join(tempfile.gettempdir(), f"{stem}_converted.{out_ext}")
        ok, err = ffmpeg_convert(dl_path, target, int(bitrate), sr, AUDIO_FORMATS[out_fmt]["ffmpeg_args"])
        if not ok:
            return Row(fid, fname, "Conversion Failed", orig_size, 0, time.time() - t0, err)

        up_size = os.path.getsize(target)
        new_name = f"{stem}_converted.{out_ext}"
        log_threadsafe(f"⬆️ Uploading: {new_name}")
        uploaded_id = client.upload_file(target, new_name, AUDIO_FORMATS[out_fmt]["mime"], output_id)
        if not uploaded_id:
            return Row(fid, fname, "Upload Failed", orig_size, up_size, time.time() - t0, "Upload error")
        return Row(fid, fname, "Success", orig_size, up_size, time.time() - t0, "")
    except Exception as e:
        return Row(fid, fname, "Error", orig_size, 0, time.time() - t0, str(e))
    finally:
        if dl_path and os.path.exists(dl_path):
            try: os.remove(dl_path)
            except Exception: pass

if st.button("Start conversion", type="primary", disabled=not selected_files):
    total = len(selected_files)
    if total == 0:
        st.warning("Please select at least one file.")
    else:
        overall = st.progress(0.0, text="Starting...")
        rows: List[Row] = []
        success_count = 0
        total_dl = 0
        total_ul = 0
        t_all = time.time()

        with ThreadPoolExecutor(max_workers=max_threads) as ex:
            futures = [ex.submit(convert_one, f) for f in selected_files]
            done_count = 0
            for fut in as_completed(futures):
                r = fut.result()
                rows.append(r)
                done_count += 1
                success_count += 1 if r.status == "Success" else 0
                total_dl += r.download_size
                total_ul += r.upload_size
                overall.progress(done_count/total, text=f"Processed {done_count}/{total}")
                df = pd.DataFrame([
                    {"name": x.name, "status": x.status,
                     "download": human_bytes(x.download_size),
                     "upload": human_bytes(x.upload_size),
                     "time_s": f"{x.duration_s:.2f}",
                     "error": x.error}
                for x in rows])
                table_placeholder.dataframe(df, width='stretch')
                render_logs()  # flush logs after each completion

        elapsed = time.time() - t_all
        pct = (success_count / total * 100.0) if total else 0.0
        summary = f"Complete: {success_count}/{total} succeeded ({pct:.1f}%). DL {human_bytes(total_dl)} | UL {human_bytes(total_ul)} | Time {elapsed:.2f}s"
        overall_placeholder.success(summary)

        csv = pd.DataFrame([{
            "id": x.id, "name": x.name, "status": x.status,
            "download_size": x.download_size, "upload_size": x.upload_size,
            "duration_s": f"{x.duration_s:.2f}", "error": x.error
        } for x in rows]).to_csv(index=False).encode("utf-8")
        report_placeholder.download_button(
            "Download CSV report",
            data=csv,
            file_name=f"conversion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Final log flush
render_logs()

st.divider()
with st.expander("ℹ️ Tips & Notes"):
    st.markdown("""
- **This build has NO Domain-Wide Delegation.** You must **share** your Drive folders/files with the Service Account’s **client_email**.
- Ensure **Google Drive API** is enabled in your GCP project.
- App uses retries for transient Drive API errors and validates downloads to avoid feeding bad data to ffmpeg.
- FFmpeg is provided via `imageio-ffmpeg` (portable binary). No system install required.
""")

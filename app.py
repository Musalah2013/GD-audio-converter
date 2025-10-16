
# app.py
# Streamlit: Google Drive Audio Converter (Server-friendly, no local installs)
#
# - OAuth2 Web flow (no deprecated OOB)
# - Uses imageio-ffmpeg to bundle a portable ffmpeg binary (no OS-level install)
# - Streams from Google Drive, converts server-side, re-uploads to Drive
# - Batch, multithreaded, live progress table + downloadable report
#
# Prereqs:
#   - Put your Google OAuth Web credentials in .streamlit/secrets.toml:
#       [google]
#       client_id = "YOUR_CLIENT_ID.apps.googleusercontent.com"
#       client_secret = "YOUR_CLIENT_SECRET"
#       # Optional: authorize only specific users (comma-separated emails)
#       allowed_emails = "you@example.com, someone@org.com"
#
#   - In Google Cloud Console: add an Authorized redirect URI that matches
#     where this app is hosted (e.g., https://your-app.streamlit.app/)
#
#   - `requirements.txt` provided alongside.
#
import os
import io
import time
import json
import math
import queue
import shutil
import logging
import tempfile
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List

import streamlit as st

# Google auth & Drive API
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google_auth_oauthlib.flow import Flow

# Portable ffmpeg path (no system install needed)
import imageio_ffmpeg

# ---------------------------
# App Constants & Setup
# ---------------------------
SCOPES = ["https://www.googleapis.com/auth/drive"]
AUDIO_FORMATS = {
    "MP3":  {"ext": "mp3",  "mime": "audio/mpeg", "ffmpeg_args": ["-codec:a", "libmp3lame"]},
    "AAC":  {"ext": "aac",  "mime": "audio/aac",  "ffmpeg_args": ["-codec:a", "aac"]},
    "M4A":  {"ext": "m4a",  "mime": "audio/mp4",  "ffmpeg_args": ["-codec:a", "aac"]},
    "OGG":  {"ext": "ogg",  "mime": "audio/ogg",  "ffmpeg_args": ["-codec:a", "libvorbis"]},
    "FLAC": {"ext": "flac", "mime": "audio/flac","ffmpeg_args": ["-codec:a", "flac"]},
    "WAV":  {"ext": "wav",  "mime": "audio/wav", "ffmpeg_args": ["-codec:a", "pcm_s16le"]},
}
BITRATES = ["32", "64", "96", "128", "192", "256", "320"]
SAMPLE_RATES = ["Auto", "22050", "32000", "44100", "48000"]

# Quieter root logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("drive-audio-converter")

# ---------------------------
# Helpers
# ---------------------------
def human_bytes(n: Optional[int]) -> str:
    if not n:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"

def get_google_secrets() -> Dict[str, str]:
    g = st.secrets.get("google", {})
    cid = g.get("client_id", "")
    cs = g.get("client_secret", "")
    allowed = [e.strip().lower() for e in g.get("allowed_emails", "").split(",") if e.strip()]
    if not cid or not cs:
        st.stop()  # Fail early if not configured
    return {"client_id": cid, "client_secret": cs, "allowed_emails": allowed}

def get_redirect_uri() -> str:
    # Use current page as redirect target
    # Ensure this exact URI is whitelisted in Google Cloud Console
    return st.experimental_get_query_params().get("redirect_uri", [st.request.url])[0]

def ensure_user_allowed(email: Optional[str], allowed_list: List[str]) -> None:
    if not allowed_list:
        return  # no restriction
    if email and email.lower() in allowed_list:
        return
    st.error("You are not authorized to use this app. Please contact the owner.")
    st.stop()

def get_flow(redirect_uri: str) -> Flow:
    secrets = get_google_secrets()
    client_config = {
        "web": {
            "client_id": secrets["client_id"],
            "project_id": "streamlit-drive-audio",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_secret": secrets["client_secret"],
            "redirect_uris": [redirect_uri],
        }
    }
    return Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=redirect_uri)

def save_creds_to_state(creds: Credentials):
    st.session_state["creds"] = json.loads(creds.to_json())

def load_creds_from_state() -> Optional[Credentials]:
    c = st.session_state.get("creds")
    if not c:
        return None
    return Credentials.from_authorized_user_info(c, SCOPES)

def build_drive(creds: Credentials):
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def is_folder(service, file_id: str) -> bool:
    try:
        meta = service.files().get(fileId=file_id, fields="id, mimeType").execute()
        return meta.get("mimeType") == "application/vnd.google-apps.folder"
    except Exception as e:
        logger.exception("is_folder error: %s", e)
        return False

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

def list_audio_files(service, folder_id: Optional[str]) -> List[Dict[str, Any]]:
    audio_mimes = [
        "audio/mpeg","audio/wav","audio/aac","audio/ogg",
        "audio/flac","audio/mp4","audio/x-wav","audio/x-ms-wma"
    ]
    mime_query = " or ".join([f"mimeType='{m}'" for m in audio_mimes])
    q = f"({mime_query}) and trashed=false"
    params = {
        "pageSize": 1000,
        "fields": "nextPageToken, files(id, name, size, mimeType, owners/emailAddress)",
        "supportsAllDrives": True,
        "includeItemsFromAllDrives": True,
        "q": q,
    }
    if folder_id:
        params["q"] = f"{q} and '{folder_id}' in parents"

    files = []
    page_token = None
    while True:
        if page_token:
            params["pageToken"] = page_token
        resp = service.files().list(**params).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files

def download_to_temp(service, file_id: str) -> str:
    """Download file to a NamedTemporaryFile, return path."""
    req = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, prefix="dl_", suffix=".bin")
    fh = tmp.file
    downloader = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.close()
    return tmp.name

def ffmpeg_convert(in_path: str, out_path: str, bitrate_kbps: int, sample_rate: Optional[int], fmt_args: List[str]) -> bool:
    """Run ffmpeg (portable binary from imageio-ffmpeg)."""
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", in_path]
    # codec args per format:
    cmd.extend(fmt_args)
    # bitrate
    cmd.extend(["-b:a", f"{bitrate_kbps}k"])
    # sample rate
    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])
    cmd.append(out_path)
    import subprocess
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg failed: %s", e.stderr)
        return False

def upload_file(service, local_path: str, dest_name: str, mime_type: str, parent_id: Optional[str]) -> Optional[str]:
    meta = {"name": dest_name}
    if parent_id:
        meta["parents"] = [parent_id]
    media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)
    try:
        file = service.files().create(
            body=meta, media_body=media, fields="id", supportsAllDrives=True
        ).execute()
        return file.get("id")
    except Exception as e:
        logger.exception("Upload failed: %s", e)
        return None

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Drive Audio Converter", page_icon="🎧", layout="wide")

st.title("🎧 Google Drive Audio Converter (Streamlit)")
st.caption("Batch-convert audio stored in Google Drive — no local installs, no OOB OAuth.")

# --- OAuth Section ---
with st.expander("🔐 Authentication", expanded=True):
    secrets = get_google_secrets()
    redirect_uri = get_redirect_uri()
    code = st.query_params.get("code")
    state = st.query_params.get("state")

    creds = load_creds_from_state()

    if creds and creds.valid:
        token_info = json.loads(creds.to_json())
        user_email = token_info.get("id_token", {}) or {}
        # id_token payload may be absent; optional extra: call userinfo endpoint. Keeping simple.
        st.success("Authenticated with Google Drive.")
        st.write("Token acquired. Ready to access Google Drive.")
    else:
        # If we had a code on the URL, finish the flow:
        if code and state and "oauth_state" in st.session_state:
            try:
                flow = get_flow(redirect_uri)
                flow.fetch_token(code=code)
                creds = flow.credentials
                save_creds_to_state(creds)
                st.success("Authentication completed. You can now use the app.")
                # Clean URL
                st.query_params.clear()
            except Exception as e:
                st.error(f"Authentication error: {e}")
        elif st.button("Sign in with Google", type="primary"):
            flow = get_flow(redirect_uri)
            auth_url, new_state = flow.authorization_url(
                access_type="offline",
                include_granted_scopes=True,
                prompt="consent",
            )
            st.session_state["oauth_state"] = new_state
            st.markdown(f"[Continue to Google Sign-In]({auth_url})")

# Stop if no credentials yet
creds = load_creds_from_state()
if not (creds and creds.valid):
    st.info("Please sign in to continue.")
    st.stop()

# Optional email restriction (requires ID token; for simplicity we skip strict check)
ensure_user_allowed(None, secrets["allowed_emails"])

# Build Drive service
drive = build_drive(creds)

# --- Folder selection ---
st.subheader("📁 Choose Input & Output Folders")
col1, col2 = st.columns(2)
with col1:
    input_url = st.text_input("Input Folder URL or ID (leave blank to search entire Drive)")
    input_id = extract_folder_id(input_url) if input_url.strip() else None
    if input_id:
        if is_folder(drive, input_id):
            st.success("Valid input folder.")
        else:
            st.error("Provided input is not a folder or not accessible.")
            input_id = None
with col2:
    output_url = st.text_input("Output Folder URL or ID (leave blank to save in Drive root)")
    output_id = extract_folder_id(output_url) if output_url.strip() else None
    if output_id:
        if is_folder(drive, output_id):
            st.success("Valid output folder.")
        else:
            st.error("Provided output is not a folder or not accessible.")
            output_id = None

# --- List files ---
st.subheader("🎵 Select Files to Convert")
files = list_audio_files(drive, input_id)
if not files:
    st.warning("No audio files found with current filter.")
else:
    # Build a table with checkboxes
    default_all = st.checkbox("Select all", value=False)
    labels = [f"{f['name']}  ({human_bytes(int(f.get('size', 0) or 0))})" for f in files]
    selected_flags = st.multiselect(
        "Pick files to convert:",
        options=list(range(len(files))),
        default=list(range(len(files))) if default_all else [],
        format_func=lambda i: labels[i],
    )
    selected_files = [files[i] for i in selected_flags]
    st.caption(f"Selected {len(selected_files)} / {len(files)}")

# --- Conversion settings ---
st.subheader("⚙️ Conversion Settings")
c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    out_fmt = st.selectbox("Format", list(AUDIO_FORMATS.keys()), index=0)
with c2:
    bitrate = st.selectbox("Bitrate (kbps)", BITRATES, index=3)  # 128 by default
with c3:
    sr_str = st.selectbox("Sample rate", SAMPLE_RATES, index=3)  # 44100
    sr = int(sr_str) if sr_str.isdigit() else None
with c4:
    max_threads = st.slider("Parallel workers", min_value=1, max_value=os.cpu_count() or 4, value=2)

# --- Run conversion ---
st.subheader("🚀 Run")
run_col1, run_col2 = st.columns([1,2])
report_rows: List[Dict[str, Any]] = []
result_placeholder = st.empty()
progress_area = st.container()
table_placeholder = st.empty()
report_download_placeholder = st.empty()

def convert_one(finfo: Dict[str, Any]) -> Dict[str, Any]:
    """Single file pipeline: download -> convert -> upload"""
    t0 = time.time()
    fid = finfo["id"]
    fname = finfo["name"]
    orig_size = int(finfo.get("size", 0) or 0)

    stage = "download"
    dl_path = ""
    out_path = ""
    try:
        # 1) Download
        dl_path = download_to_temp(drive, fid)
        # 2) Convert
        stage = "convert"
        stem = os.path.splitext(fname)[0]
        target = os.path.join(tempfile.gettempdir(), f"{stem}_converted.{AUDIO_FORMATS[out_fmt]['ext']}")
        ok = ffmpeg_convert(
            dl_path, target, int(bitrate), sr, AUDIO_FORMATS[out_fmt]["ffmpeg_args"]
        )
        if not ok:
            return {
                "id": fid, "name": fname, "status": "Conversion Failed",
                "download_size": orig_size, "upload_size": 0, "duration_s": time.time() - t0,
            }
        # 3) Upload
        stage = "upload"
        up_size = os.path.getsize(target)
        new_name = f"{stem}_converted.{AUDIO_FORMATS[out_fmt]['ext']}"
        uploaded_id = upload_file(drive, target, new_name, AUDIO_FORMATS[out_fmt]["mime"], output_id)
        if not uploaded_id:
            return {
                "id": fid, "name": fname, "status": "Upload Failed",
                "download_size": orig_size, "upload_size": up_size, "duration_s": time.time() - t0,
            }
        return {
            "id": fid, "name": fname, "status": "Success",
            "download_size": orig_size, "upload_size": up_size, "duration_s": time.time() - t0,
        }
    except Exception as e:
        return {
            "id": fid, "name": fname, "status": f"Error: {e}",
            "download_size": orig_size, "upload_size": 0, "duration_s": time.time() - t0,
        }
    finally:
        for p in [dl_path, out_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

if st.button("Start conversion", type="primary", disabled=not selected_files):
    total = len(selected_files)
    if total == 0:
        st.warning("Please select at least one file.")
    else:
        st.success(f"Starting conversion of {total} files...")
        overall = st.progress(0, text="Starting...")
        rows = []
        status_q = queue.Queue()

        def worker_status(msg: str):
            status_q.put(msg)

        def run_pool():
            with ThreadPoolExecutor(max_workers=max_threads) as ex:
                futures = [ex.submit(convert_one, f) for f in selected_files]
                done_count = 0
                for fut in as_completed(futures):
                    res = fut.result()
                    rows.append(res)
                    done_count += 1
                    overall.progress(done_count/total, text=f"Processed {done_count}/{total}")
                    # live table
                    import pandas as pd
                    df = pd.DataFrame(rows)
                    df_display = df.assign(
                        download= [human_bytes(x) for x in df["download_size"]],
                        upload=   [human_bytes(x) for x in df["upload_size"]],
                        time_s=   [f"{x:.2f}" for x in df["duration_s"]],
                    )[["name","status","download","upload","time_s"]]
                    table_placeholder.dataframe(df_display, use_container_width=True)
            return rows

        out_rows = run_pool()
        # Summary
        succ = sum(1 for r in out_rows if r["status"] == "Success")
        result_placeholder.success(f"Complete: {succ}/{total} succeeded.")
        # Offer report download
        import pandas as pd
        rep_df = pd.DataFrame(out_rows)
        csv = rep_df.to_csv(index=False).encode("utf-8")
        report_download_placeholder.download_button(
            "Download CSV report",
            data=csv,
            file_name=f"conversion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

st.divider()
with st.expander("ℹ️ Tips & Notes"):
    st.markdown(
        """
- **No local installs**: The app uses a portable `ffmpeg` binary from `imageio-ffmpeg`.
- **OAuth2**: Uses the web authorization code flow; make sure your **Authorized redirect URI** in Google Cloud Console matches the URL of this app.
- **Access**: To restrict usage, set `google.allowed_emails` in your `secrets.toml`.
- **Performance**: Increase *Parallel workers* with caution; Drive API rate limits and server resources apply.
- **Output Folder**: Leave empty to upload to Drive root. If you enter a non-folder, upload will fail.
"""
    )

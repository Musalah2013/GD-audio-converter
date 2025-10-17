
# app_service_account_upload.py
# Streamlit: Google Drive Audio Converter (Upload SA JSON per session)
#
# - User uploads a Service Account JSON on each use (no secrets required)
# - Optional Domain-Wide Delegation (checkbox + email field)
# - No local FFmpeg install required (uses imageio-ffmpeg portable binary)
# - Batch conversion with parallel workers, live progress, CSV report
#
import os
import io
import time
import json
import tempfile
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account

import imageio_ffmpeg
import subprocess

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("drive-audio-converter-upload")

st.set_page_config(page_title="Drive Audio Converter (Upload SA)", page_icon="🗝️", layout="wide")
st.title("🗝️ Google Drive Audio Converter — Upload Service Account JSON")
st.caption("Upload a Google Service Account JSON on each session. No project secrets needed.")

# ---------------------------
# Auth via uploaded SA JSON
# ---------------------------
with st.expander("🔐 Upload Service Account JSON", expanded=True):
    sa_file = st.file_uploader("Upload your Service Account JSON key", type=["json"])
    use_dwd = st.checkbox("Use Domain‑Wide Delegation (impersonate a user)?", value=False,
                          help="Requires Workspace admin configuration for DWD.")
    subject = st.text_input("Impersonated user email (only if using DWD)", value="", disabled=not use_dwd)
    build_client = st.button("Build Drive client", type="primary")

def build_drive(sa_json_text: str, use_dwd: bool, subject: str):
    try:
        info = json.loads(sa_json_text)
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    if use_dwd:
        if not subject.strip():
            st.error("Please provide an email to impersonate when using DWD.")
            st.stop()
        creds = creds.with_subject(subject.strip())
    svc = build("drive", "v3", credentials=creds, cache_discovery=False)
    who = info.get("client_email", "(unknown)")
    st.success(f"Authenticated with Service Account: **{who}**" + (f" — impersonating **{subject.strip()}**" if use_dwd else ""))
    return svc

if "drive" not in st.session_state and build_client:
    if not sa_file:
        st.error("Please upload a Service Account JSON first.")
    else:
        st.session_state["sa_json"] = sa_file.read().decode("utf-8")
        st.session_state["drive"] = build_drive(st.session_state["sa_json"], use_dwd, subject)

if "drive" not in st.session_state:
    st.info("Upload your Service Account JSON and click **Build Drive client** to continue.")
    st.stop()

drive = st.session_state["drive"]

# ---------------------------
# Utility functions
# ---------------------------
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

def is_folder(service, file_id: str) -> bool:
    try:
        meta = service.files().get(fileId=file_id, fields="id, mimeType", supportsAllDrives=True).execute()
        return meta.get("mimeType") == "application/vnd.google-apps.folder"
    except Exception as e:
        logger.exception("is_folder error: %s", e)
        return False

def list_audio_files(service, folder_id: Optional[str]) -> List[Dict[str, Any]]:
    audio_mimes = [
        "audio/mpeg","audio/wav","audio/aac","audio/ogg",
        "audio/flac","audio/mp4","audio/x-wav","audio/x-ms-wma"
    ]
    mime_query = " or ".join([f"mimeType='{m}'" for m in audio_mimes])
    q = f"({mime_query}) and trashed=false"
    params = {
        "pageSize": 1000,
        "fields": "nextPageToken, files(id, name, size, mimeType, parents)",
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

def download_to_temp(service, file_id: str) -> str:
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
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", in_path]
    cmd.extend(fmt_args)
    cmd.extend(["-b:a", f"{bitrate_kbps}k"])
    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])
    cmd.append(out_path)
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
# UI: Folders, Files, Settings
# ---------------------------
st.subheader("📁 Choose Input & Output Folders")
c1, c2 = st.columns(2)
with c1:
    input_url = st.text_input("Input Folder URL or ID (must be accessible to this SA or the impersonated user)")
    input_id = extract_folder_id(input_url) if input_url.strip() else None
    if input_id:
        if is_folder(drive, input_id):
            st.success("Valid input folder.")
        else:
            st.error("Provided input is not a folder or not accessible.")
            input_id = None
with c2:
    output_url = st.text_input("Output Folder URL or ID (leave blank to save in Drive root)")
    output_id = extract_folder_id(output_url) if output_url.strip() else None
    if output_id:
        if is_folder(drive, output_id):
            st.success("Valid output folder.")
        else:
            st.error("Provided output is not a folder or not accessible.")
            output_id = None

st.subheader("🎵 Select Files to Convert")
files = list_audio_files(drive, input_id)
if not files:
    st.warning("No audio files found. Ensure the folder is shared with the Service Account or that DWD is correctly configured.")
    selected_files = []
else:
    labels = [f"{f['name']}  ({human_bytes(int(f.get('size', 0) or 0))})" for f in files]
    default_all = st.checkbox("Select all", value=False)
    selected_flags = st.multiselect(
        "Pick files to convert:",
        options=list(range(len(files))),
        default=list(range(len(files))) if default_all else [],
        format_func=lambda i: labels[i],
    )
    selected_files = [files[i] for i in selected_flags]
    st.caption(f"Selected {len(selected_files)} / {len(files)}")

st.subheader("⚙️ Conversion Settings")
d1, d2, d3, d4 = st.columns([1,1,1,1])
with d1:
    out_fmt = st.selectbox("Format", list(AUDIO_FORMATS.keys()), index=0)
with d2:
    bitrate = st.selectbox("Bitrate (kbps)", BITRATES, index=3)
with d3:
    sr_str = st.selectbox("Sample rate", SAMPLE_RATES, index=3)
    sr = int(sr_str) if sr_str.isdigit() else None
with d4:
    max_threads = st.slider("Parallel workers", min_value=1, max_value=os.cpu_count() or 4, value=2)

# ---------------------------
# Run conversion
# ---------------------------
st.subheader("🚀 Run")
result_placeholder = st.empty()
table_placeholder = st.empty()
report_download_placeholder = st.empty()

def convert_one(finfo: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    fid = finfo["id"]
    fname = finfo["name"]
    orig_size = int(finfo.get("size", 0) or 0)
    dl_path = ""
    try:
        dl_path = download_to_temp(drive, fid)
        stem = os.path.splitext(fname)[0]
        target = os.path.join(tempfile.gettempdir(), f"{stem}_converted.{AUDIO_FORMATS[out_fmt]['ext']}")
        ok = ffmpeg_convert(dl_path, target, int(bitrate), sr, AUDIO_FORMATS[out_fmt]["ffmpeg_args"])
        if not ok:
            return {"id": fid, "name": fname, "status": "Conversion Failed",
                    "download_size": orig_size, "upload_size": 0, "duration_s": time.time() - t0}
        up_size = os.path.getsize(target)
        new_name = f"{stem}_converted.{AUDIO_FORMATS[out_fmt]['ext']}"
        uploaded_id = upload_file(drive, target, new_name, AUDIO_FORMATS[out_fmt]["mime"], output_id)
        if not uploaded_id:
            return {"id": fid, "name": fname, "status": "Upload Failed",
                    "download_size": orig_size, "upload_size": up_size, "duration_s": time.time() - t0}
        return {"id": fid, "name": fname, "status": "Success",
                "download_size": orig_size, "upload_size": up_size, "duration_s": time.time() - t0}
    except Exception as e:
        return {"id": fid, "name": fname, "status": f"Error: {e}",
                "download_size": orig_size, "upload_size": 0, "duration_s": time.time() - t0}
    finally:
        if dl_path and os.path.exists(dl_path):
            try: os.remove(dl_path)
            except Exception: pass

if st.button("Start conversion", type="primary", disabled=not selected_files):
    total = len(selected_files)
    if total == 0:
        st.warning("Please select at least one file.")
    else:
        st.success(f"Starting conversion of {total} files...")
        overall = st.progress(0, text="Starting...")
        rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_threads) as ex:
            futures = [ex.submit(convert_one, f) for f in selected_files]
            done_count = 0
            for fut in as_completed(futures):
                res = fut.result()
                rows.append(res)
                done_count += 1
                overall.progress(done_count/total, text=f"Processed {done_count}/{total}")
                df = pd.DataFrame(rows).assign(
                    download = lambda d: d["download_size"].map(human_bytes),
                    upload   = lambda d: d["upload_size"].map(human_bytes),
                    time_s   = lambda d: d["duration_s"].map(lambda x: f"{x:.2f}")
                )[["name","status","download","upload","time_s"]]
                table_placeholder.dataframe(df, use_container_width=True)
        succ = sum(1 for r in rows if r["status"] == "Success")
        result_placeholder.success(f"Complete: {succ}/{total} succeeded.")
        csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        report_download_placeholder.download_button(
            "Download CSV report",
            data=csv,
            file_name=f"conversion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

st.divider()
with st.expander("ℹ️ Tips & Notes"):
    st.markdown("""
- **Upload SA JSON**: The key is held in memory for this session only; it is **not** written to disk by the app.
- **Domain-Wide Delegation**: Tick the checkbox and provide an email to impersonate. Requires Workspace DWD configuration.
- **Shared Drives**: Make sure the SA (or impersonated user) has access to those folders.
- **No local FFmpeg**: Uses `imageio-ffmpeg` for a portable binary.
""")

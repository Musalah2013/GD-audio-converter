# ============================================================================
# CRITICAL: Set env vars FIRST, before any imports
# ============================================================================
import os
import sys

# Disable PyArrow entirely to prevent segfaults in slim containers
os.environ["STREAMLIT_DATAFRAME_SERIALIZATION"] = "legacy"
os.environ["PANDAS_USE_PYARROW"] = "0"
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"  # Extra safety

# Force pandas to use object dtype instead of nullable dtypes (which can trigger Arrow)
os.environ["PANDAS_NULLABLE_DTYPES_DISABLED"] = "1"

# Prevent any PyArrow import attempts
sys.modules["pyarrow"] = None

# ============================================================================
# NOW safe to import everything else
# ============================================================================
import time
import json
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

import streamlit as st
import pandas as pd

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from google.auth.transport.requests import Request, AuthorizedSession
from google.auth import exceptions as ga_exceptions

import imageio_ffmpeg

# ... rest of code unchanged ...


# Streamlit – Google Drive Audio Converter

## Quick Start

1. Create a Google Cloud "OAuth 2.0 Client ID" (type: **Web application**).
   - Add your Streamlit app URL to **Authorized redirect URIs** (e.g., `https://yourapp.streamlit.app/`).
2. In your Streamlit project, create `.streamlit/secrets.toml`:

```
[google]
client_id = "YOUR_CLIENT_ID.apps.googleusercontent.com"
client_secret = "YOUR_CLIENT_SECRET"
# Optional
allowed_emails = "you@example.com, colleague@org.com"
```

3. Deploy with the included `requirements.txt`.

## Notes

- No system-level ffmpeg install is needed; we bundle portable binary via `imageio-ffmpeg`.
- Conversion runs server-side, streaming to/from Google Drive.
- The app supports batch conversion, parallel workers, and CSV reporting.

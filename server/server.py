# server.py
"""
GPTCap Relay Server
- Usage: python server.py
- Make sure .env contains OPENAI_API_KEY
- Endpoints:
    GET  /                 -> status
    POST /listen           -> multipart form: file=@your_audio.wav
    GET  /audio/<filename> -> serves TTS audio file
    GET  /reply            -> returns last reply (json)
    POST /photo            -> multipart form: file=@photo.jpg
"""
import os
import uuid
import time
import shutil
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import mimetypes
import base64

# load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing from environment (.env)")

# config (overridable via .env)
PORT = int(os.getenv("PORT", 5000))
TMP_DIR = Path(os.getenv("TMP_DIR", "/tmp/gptcap"))
TMP_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
VOICE = os.getenv("VOICE", "alloy")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "8"))
ENABLE_VISION = os.getenv("ENABLE_VISION", "false").lower() in ("1", "true", "yes")

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gptcap-server")

# Try to instantiate latest OpenAI client (newer openai-python lib)
USING_OPENAI_CLIENT = False
openai_client = None
try:
    # new-style client (openai>=1.0.0 variants)
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    USING_OPENAI_CLIENT = True
    logger.info("Using OpenAI OpenAI client (new-style).")
except Exception as e:
    try:
        # older-style openai package (set global api_key)
        import openai as openai_legacy
        openai_legacy.api_key = OPENAI_API_KEY
        openai_client = openai_legacy
        USING_OPENAI_CLIENT = False
        logger.info("Using legacy openai package (global api_key).")
    except Exception as e2:
        logger.warning("OpenAI package import failed; will fallback to direct REST calls.")
        openai_client = None
        USING_OPENAI_CLIENT = False

app = Flask(__name__)
CORS(app)  # allow dashboard (mobile) to poll

# in-memory last reply store (simple)
LAST_REPLY = {"text": "", "audio": ""}

# helper: enforce max upload size
def check_content_length():
    length = request.content_length
    if length is not None and length > MAX_UPLOAD_BYTES:
        abort(413, description=f"Upload too large (>{MAX_UPLOAD_MB} MB)")

# helper to safe-write streaming content
def write_streamed_response_to_file(stream_resp, dest_path):
    with open(dest_path, "wb") as out_f:
        for chunk in stream_resp.iter_content(chunk_size=8192):
            if chunk:
                out_f.write(chunk)

@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "note": "GPTCap relay server",
        "models": {"transcribe": TRANSCRIBE_MODEL, "chat": CHAT_MODEL, "tts": TTS_MODEL},
        "enable_vision": ENABLE_VISION
    })

@app.route("/listen", methods=["POST"])
def listen():
    """
    Expect multipart form with 'file' field containing a small WAV/MP3 etc.
    Returns JSON: {transcript, reply, audio_url}
    """
    check_content_length()
    if 'file' not in request.files:
        return jsonify({"error": "Missing 'file' field"}), 400

    f = request.files['file']
    ext = Path(f.filename).suffix or ".wav"
    tmp_audio_name = f"{uuid.uuid4()}{ext}"
    tmp_audio_path = TMP_DIR / tmp_audio_name
    f.save(tmp_audio_path)
    logger.info("Saved upload to %s", str(tmp_audio_path))

    try:
        # --------- 1) Transcribe -----------
        user_text = None
        # Prefer SDK if available
        if USING_OPENAI_CLIENT and hasattr(openai_client, "audio"):
            try:
                with open(tmp_audio_path, "rb") as audio_fd:
                    # SDK interfaces differ between versions — try common patterns
                    if hasattr(openai_client.audio, "transcriptions"):
                        trans = openai_client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=audio_fd)
                        user_text = getattr(trans, "text", trans.get("text") if isinstance(trans, dict) else None)
                    elif hasattr(openai_client.audio, "transcribe"):
                        trans = openai_client.audio.transcribe(model=TRANSCRIBE_MODEL, file=audio_fd)
                        user_text = getattr(trans, "text", trans.get("text") if isinstance(trans, dict) else None)
            except Exception as e:
                logger.warning("SDK transcription attempt failed: %s", e)
        # REST fallback
        if not user_text:
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            files = {"file": open(tmp_audio_path, "rb")}
            data = {"model": TRANSCRIBE_MODEL}
            r = requests.post(url, headers=headers, files=files, data=data, timeout=60)
            r.raise_for_status()
            user_text = r.json().get("text")

        if not user_text:
            user_text = "(untranscribed)"

        logger.info("Transcribed text: %s", user_text)

        # --------- 2) Chat with model -----------
        reply_text = ""
        if USING_OPENAI_CLIENT and hasattr(openai_client, "chat") and hasattr(openai_client.chat, "completions"):
            try:
                chat_resp = openai_client.chat.completions.create(
                    model=CHAT_MODEL, messages=[{"role": "user", "content": user_text}], max_tokens=512
                )
                # supporting different return formats
                first_choice = chat_resp.choices[0]
                reply_text = getattr(first_choice.message, "content", None) or first_choice.get("message", {}).get("content")
            except Exception as e:
                logger.warning("SDK chat attempt failed: %s", e)

        if not reply_text:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {"model": CHAT_MODEL, "messages": [{"role": "user", "content": user_text}], "max_tokens": 512}
            r2 = requests.post(url, headers=headers, json=payload, timeout=60)
            r2.raise_for_status()
            reply_text = r2.json()["choices"][0]["message"]["content"]

        logger.info("Model replied: %s", reply_text)

        # --------- 3) TTS (reply -> MP3) ----------
        tts_filename = f"{uuid.uuid4()}.mp3"
        tts_path = TMP_DIR / tts_filename
        tts_written = False

        if USING_OPENAI_CLIENT and hasattr(openai_client, "audio") and hasattr(openai_client.audio, "speech"):
            try:
                # SDK-style TTS (names vary) - attempt common pattern
                audio_resp = openai_client.audio.speech.create(model=TTS_MODEL, voice=VOICE, input=reply_text)
                # SDK might give streaming interface or raw bytes
                if hasattr(audio_resp, "stream_to_file"):
                    audio_resp.stream_to_file(str(tts_path))
                    tts_written = True
                elif isinstance(audio_resp, (bytes, bytearray)):
                    with open(tts_path, "wb") as f_out:
                        f_out.write(audio_resp)
                    tts_written = True
                else:
                    # If it's file-like: try saving bytes
                    try:
                        with open(tts_path, "wb") as f_out:
                            f_out.write(audio_resp.read())
                        tts_written = True
                    except Exception:
                        logger.warning("SDK TTS produced unexpected type, falling back to REST.")
            except Exception as e:
                logger.warning("SDK TTS attempt failed: %s", e)

        if not tts_written:
            # REST fallback: audio/speech
            tts_url = "https://api.openai.com/v1/audio/speech"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": TTS_MODEL, "voice": VOICE, "input": reply_text}
            # stream writing
            r3 = requests.post(tts_url, headers=headers, json=payload, stream=True, timeout=120)
            r3.raise_for_status()
            write_streamed_response_to_file(r3, tts_path)
            tts_written = True

        # save last reply info
        LAST_REPLY["text"] = reply_text
        LAST_REPLY["audio"] = tts_filename

        audio_url = f"/audio/{tts_filename}"
        return jsonify({"transcript": user_text, "reply": reply_text, "audio_url": audio_url})

    except requests.HTTPError as he:
        logger.exception("HTTP error during /listen pipeline")
        return jsonify({"error": "Upstream HTTP error", "detail": str(he)}), 502
    except Exception as e:
        logger.exception("Error in /listen")
        return jsonify({"error": str(e)}), 500
    finally:
        # optional: remove uploaded audio (keep TTS for client to fetch)
        try:
            tmp_audio_path.unlink(missing_ok=True)
        except Exception:
            pass

@app.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    # serve TTS files
    p = TMP_DIR / filename
    if not p.exists():
        abort(404)
    # guess mimetype
    mime, _ = mimetypes.guess_type(str(p))
    return send_file(str(p), mimetype=mime or "application/octet-stream", as_attachment=False)

@app.route("/reply", methods=["GET"])
def get_last_reply():
    # dashboard polling
    return jsonify(LAST_REPLY)

@app.route("/photo", methods=["POST"])
def photo_upload():
    """
    Save uploaded image. If ENABLE_VISION is true, attempt to ask model to describe image.
    Note: for vision to work reliably the model may need a public URL to fetch the image; some SDKs accept base64 inline images.
    """
    check_content_length()
    if 'file' not in request.files:
        return jsonify({"error": "Missing 'file' field"}), 400

    f = request.files['file']
    ext = Path(f.filename).suffix or ".jpg"
    photo_name = f"{uuid.uuid4()}{ext}"
    photo_path = TMP_DIR / photo_name
    f.save(photo_path)
    logger.info("Saved photo %s", str(photo_path))

    result_description = "saved"
    # Attempt vision analysis if enabled
    if ENABLE_VISION:
        try:
            # Read and base64-encode the image as data URL
            with open(photo_path, "rb") as imgf:
                b64 = base64.b64encode(imgf.read()).decode()
            data_url = f"data:image/{ext.lstrip('.')};base64,{b64}"

            # Try SDK vision path
            vision_reply = None
            if USING_OPENAI_CLIENT:
                try:
                    # Many SDKs accept messages with image content or use a vision-specific endpoint.
                    # We'll attempt a chat completion with the image embedded as a data URL message (may or may not be supported depending on SDK/version)
                    prompt_messages = [
                        {"role": "user", "content": "Describe this image in one short sentence."},
                        {"role": "user", "content": data_url}
                    ]
                    vc = openai_client.chat.completions.create(model=CHAT_MODEL, messages=prompt_messages, max_tokens=250)
                    vision_reply = vc.choices[0].message["content"]
                except Exception as e:
                    logger.warning("SDK vision attempt failed: %s", e)

            # REST fallback: attempt to call chat completions with the base64 data in the messages
            if not vision_reply:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": CHAT_MODEL,
                    "messages": [
                        {"role": "user", "content": "Describe this image in one short sentence."},
                        {"role": "user", "content": data_url}
                    ],
                    "max_tokens": 250
                }
                r = requests.post(url, headers=headers, json=payload, timeout=60)
                r.raise_for_status()
                vision_reply = r.json()["choices"][0]["message"]["content"]

            result_description = vision_reply or "no description"
            # update last reply
            LAST_REPLY["text"] = f"[vision] {result_description}"
        except Exception as e:
            logger.exception("Vision attempt failed")
            result_description = f"vision_failed: {e}"

    return jsonify({"status": "ok", "photo": f"/photos/{photo_name}", "description": result_description})

@app.route("/photos/<filename>", methods=["GET"])
def serve_photo(filename):
    p = TMP_DIR / filename
    if not p.exists():
        abort(404)
    mime, _ = mimetypes.guess_type(str(p))
    return send_file(str(p), mimetype=mime or "image/jpeg", as_attachment=False)

if __name__ == "__main__":
    logger.info("Starting GPTCap relay server on port %s", PORT)
    # dev server only: use gunicorn/uvicorn in production
    app.run(host="0.0.0.0", port=PORT, debug=True)

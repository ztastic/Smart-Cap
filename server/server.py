import os, uuid, base64, mimetypes, json, re
from pathlib import Path
from flask import Flask, request, jsonify, send_file, abort
import logging, requests

# --- Config ---
PORT = int(os.getenv("PORT", 5000))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
VOICE = os.getenv("VOICE", "alloy")
ENABLE_VISION = os.getenv("ENABLE_VISION", "0") == "1"

TMP_DIR = Path("tmp")
AUDIO_OUT_DIR = TMP_DIR
TMP_DIR.mkdir(exist_ok=True)

LAST_REPLY = {"text": None, "audio": None}

# Try SDK client
USING_OPENAI_CLIENT = False
openai_client = None
try:
    import openai
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    USING_OPENAI_CLIENT = True
except Exception as e:
    print("No SDK client, fallback only:", e)

# Flask app
app = Flask(__name__)
logger = logging.getLogger("gptcap-server")
logging.basicConfig(level=logging.INFO)

def write_streamed_response_to_file(resp, out_path):
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def check_content_length(max_size=20*1024*1024):
    cl = request.content_length
    if cl is not None and cl > max_size:
        abort(413, f"File too large ({cl} > {max_size})")

@app.route("/listen", methods=["POST"])
def listen():
    check_content_length()

    audio_file = request.files.get("file")
    image_file = request.files.get("image")

    if not audio_file:
        return jsonify({"error": "Missing 'file' (audio) field"}), 400

    # --- Save audio ---
    ext = Path(audio_file.filename).suffix or ".wav"
    tmp_audio_path = TMP_DIR / f"{uuid.uuid4()}{ext}"
    audio_file.save(tmp_audio_path)
    logger.info("Saved audio upload to %s", str(tmp_audio_path))

    # --- Save image (if provided) ---
    image_path = None
    image_data_url = None
    if image_file:
        try:
            img_ext = Path(image_file.filename).suffix or ".jpg"
            image_path = TMP_DIR / f"{uuid.uuid4()}{img_ext}"
            image_file.save(image_path)
            logger.info("Saved image upload to %s", str(image_path))

            # Base64 encode for GPT vision input
            with open(image_path, "rb") as imgfd:
                b64 = base64.b64encode(imgfd.read()).decode()
            image_data_url = f"data:image/{img_ext.lstrip('.')};base64,{b64}"
        except Exception as e:
            logger.exception("Failed saving/encoding uploaded image")
            image_path = None
            image_data_url = None

    # --------- 1) Transcribe ----------
    user_text = None
    try:
        with open(tmp_audio_path, "rb") as audio_fd:
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            files = {"file": audio_fd}
            data = {"model": TRANSCRIBE_MODEL}
            r = requests.post(url, headers=headers, files=files, data=data, timeout=60)
            r.raise_for_status()
            user_text = r.json().get("text")
    except Exception:
        logger.exception("Transcription failed")
        user_text = "(untranscribed)"

    logger.info("Transcribed text: %s", user_text)

    # --------- 2) Chat with model ----------
    reply_text = ""
    command_obj = None

    system_prompt = """You are ThinkCap, a voice assistant that can also control the device.
1) If the user asks for music playback or device actions, respond in natural language
AND also output a JSON command object at the very end of your message.
Format example:
{
  "action": "play_music",
  "track": "example-song"
}
2) If the user asks the assistant to take or analyze a photo, The image will be provided to you. Use it to answer the user's question.
3) If no action is required, do not output JSON.
"""

    try:
        # --- Build message payload ---
        if image_data_url:
            # Proper multimodal input (text + image)
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text or "(no text detected)"},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }
            ]
        else:
            # Text-only input
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text or "(no text detected)"}]
                }
            ]

        # --- Send to OpenAI API ---
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": CHAT_MODEL,
            "messages": messages,
            "max_tokens": 512
        }

        r2 = requests.post(url, headers=headers, json=payload, timeout=90)
        r2.raise_for_status()
        data = r2.json()
        reply_text = data["choices"][0]["message"]["content"]

    except Exception:
        logger.exception("Chat completion failed")
        reply_text = "(error generating reply)"

    logger.info("Model replied raw: %s", reply_text)


    # Extract JSON command if present
    match = re.search(r"\{[\s\S]*\}", reply_text)  # more robust JSON capture
    if match:
        try:
            command_obj = json.loads(match.group(0))
            # remove JSON from reply_text for cleaner TTS
            reply_text = (reply_text[:match.start()] + reply_text[match.end():]).strip()
        except Exception as e:
            logger.warning("Failed to parse command JSON: %s", e)
            command_obj = None

    logger.info("Model reply text: %s", reply_text)
    if command_obj:
        logger.info("Model command: %s", command_obj)

    # --------- 3) TTS (reply -> MP3) ----------
    tts_filename = f"{uuid.uuid4()}.mp3"
    tts_path = TMP_DIR / tts_filename
    try:
        tts_url = "https://api.openai.com/v1/audio/speech"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": TTS_MODEL, "voice": VOICE, "input": reply_text}
        r3 = requests.post(tts_url, headers=headers, json=payload, stream=True, timeout=120)
        r3.raise_for_status()
        write_streamed_response_to_file(r3, tts_path)
    except Exception:
        logger.exception("TTS failed")

    audio_url = f"/audio/{tts_filename}"

    # --------- 4) Music handling ----------
    music_url = None
    if command_obj and command_obj.get("action") == "play_music":
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
        except Exception:
            logger.warning("pygame init failed for music playback")

        track = command_obj.get("track", "")
        safe_name = track.lower().replace(" ", "_") + ".mp3"
        music_path = Path("music") / safe_name

        if music_path.exists():
            logger.info("Playing track: %s", music_path)
            music_url = f"/music/{safe_name}"
            try:
                import pygame
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                pygame.mixer.music.load(str(music_path))
                pygame.mixer.music.play()
            except Exception:
                logger.exception("Error playing music")
        else:
            logger.warning("Track not found: %s", music_path)

    # Update LAST_REPLY for dashboard
    LAST_REPLY["text"] = reply_text
    LAST_REPLY["audio"] = audio_url

    # --------- 5) Return JSON response ----------
    return jsonify({
        "transcript": user_text,
        "reply": reply_text,
        "audio_url": audio_url,
        "music_url": music_url,
        "command": command_obj,
        "photo": f"/photos/{image_path.name}" if image_path else None
    })

@app.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    p = TMP_DIR / filename
    if not p.exists():
        abort(404)
    mime, _ = mimetypes.guess_type(str(p))
    return send_file(str(p), mimetype=mime or "application/octet-stream", as_attachment=False)

@app.route("/music/<filename>", methods=["GET"])
def serve_music(filename):
    p = Path("music") / filename
    if not p.exists():
        abort(404)
    mime, _ = mimetypes.guess_type(str(p))
    return send_file(str(p), mimetype=mime or "audio/mpeg", as_attachment=False)

@app.route("/reply", methods=["GET"])
def get_last_reply():
    return jsonify(LAST_REPLY)

@app.route("/vision", methods=["POST"])
def vision():
    """
    Accepts:
      - files['file'] (image)
      - form 'prompt' (optional) - the user's question about the image

    Returns:
      { reply: str, audio_url: "/audio/<file>", photo: "/photos/<file>" }
    """
    check_content_length(max_size=8*1024*1024)  # 8 MB default
    if 'file' not in request.files:
        return jsonify({"error": "Missing 'file' field"}), 400

    prompt = request.form.get("prompt", "Describe this image.")
    f = request.files['file']
    ext = Path(f.filename).suffix or ".jpg"
    photo_name = f"{uuid.uuid4()}{ext}"
    photo_path = TMP_DIR / photo_name
    f.save(photo_path)
    logger.info("Saved vision photo %s", str(photo_path))

    # read and base64 encode (safe small images)
    try:
        with open(photo_path, "rb") as imgf:
            b64 = base64.b64encode(imgf.read()).decode()
        data_url = f"data:image/{ext.lstrip('.')};base64,{b64}"
    except Exception as e:
        logger.exception("Failed to read/encode image")
        return jsonify({"error": "failed to process image"}), 500

    if photo_path.exists():
        messages = [
        {"role": "user", "content": [
            {"type": "text", "text": transcript_text},
            {"type": "image_url", "image_url": f"file://{image_path}"}
        ]}
    ]
    else:
        messages = [
        {"role": "user", "content": [
            {"type": "text", "text": transcript_text}
        ]}
    ]

    reply_text = "(vision error)"
    try:
        # Chat completion (REST)
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": CHAT_MODEL,
            "messages": messages,
            "max_tokens": 512
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        reply_text = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.exception("Vision chat failed")
        reply_text = "(vision analysis failed)"

    # TTS the reply (same as /listen TTS flow)
    tts_filename = f"{uuid.uuid4()}.mp3"
    tts_path = TMP_DIR / tts_filename
    try:
        tts_url = "https://api.openai.com/v1/audio/speech"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        tts_payload = {"model": TTS_MODEL, "voice": VOICE, "input": reply_text}
        r_tts = requests.post(tts_url, headers=headers, json=tts_payload, stream=True, timeout=120)
        r_tts.raise_for_status()
        write_streamed_response_to_file(r_tts, tts_path)
    except Exception as e:
        logger.exception("Vision TTS failed")
        # if TTS fails, we still return the textual reply

    # return JSON with a link to the saved photo and audio
    return jsonify({
        "reply": reply_text,
        "audio_url": f"/audio/{tts_filename}" if tts_path.exists() else None,
        "photo": f"/photos/{photo_name}"
    })


@app.route("/photo", methods=["POST"])
def photo_upload():
    check_content_length()
    if 'file' not in request.files:
        return jsonify({"error": "Missing 'file' field"}), 400
    f = request.files['file']
    ext = Path(f.filename).suffix or ".jpg"
    photo_name = f"{uuid.uuid4()}{ext}"
    photo_path = TMP_DIR / photo_name
    f.save(photo_path)
    logger.info("Saved photo %s", str(photo_path))
    return jsonify({"status": "ok", "photo": f"/photos/{photo_name}", "description": "saved"})

@app.route("/photos/<filename>", methods=["GET"])
def serve_photo(filename):
    p = TMP_DIR / filename
    if not p.exists():
        abort(404)
    mime, _ = mimetypes.guess_type(str(p))
    return send_file(str(p), mimetype=mime or "image/jpeg", as_attachment=False)

if __name__ == "__main__":
    logger.info("Starting GPTCap relay server on port %s", PORT)
    app.run(host="0.0.0.0", port=PORT, debug=True)

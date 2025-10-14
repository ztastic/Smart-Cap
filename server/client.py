import sounddevice as sd
import numpy as np
import requests
import json
import time
import pygame
import os
import uuid
import threading
import cv2
from scipy.io.wavfile import write
from pathlib import Path

# --- Configuration ---
SERVER_URL = "http://127.0.0.1:5000/listen"
SAMPLE_RATE = 16000
capture_audio_file = f"capture_{uuid.uuid4().hex}.wav"
capture_image_file = f"image_{uuid.uuid4().hex}.jpg"

# --- Recording state ---
recording = []
recording_active = False

# --- Audio recording function ---
def record_audio():
    global recording_active, recording
    print("🎙️ Recording... (press Enter again to stop)")
    recording_active = True
    rec = []

    def callback(indata, frames, time_info, status):
        if recording_active:
            rec.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=callback):
        while recording_active:
            sd.sleep(100)

    recording = np.concatenate(rec, axis=0)
    print(" Recording stopped.")

# --- Start recording ---
input("Press Enter to start recording...")
thread = threading.Thread(target=record_audio)
thread.start()

input("Press Enter to stop recording...")
recording_active = False
thread.join()

# --- Save the audio file ---
write(capture_audio_file, SAMPLE_RATE, recording)
print(f"Saved recording to {capture_audio_file}")

# --- Capture image from webcam ---
def capture_image():
    print("Capturing image from webcam...")
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not access webcam.")
        return None

    ret, frame = cam.read()
    cam.release()

    if not ret:
        print("Failed to capture image.")
        return None

    cv2.imwrite(capture_image_file, frame)
    print(f"Saved image to {capture_image_file}")
    return capture_image_file

image_path = capture_image()

# --- Prepare files for upload ---
files = {
    'file': open(capture_audio_file, 'rb'),
    'image': open(capture_image_file, 'rb')
}
print(f"Uploading files → {SERVER_URL} ...")
response = requests.post(SERVER_URL, files=files)
result = response.json()

print("\n--- SERVER RESPONSE ---")
print("Keys:", list(result.keys()))
print("Reply text:\n", result.get("reply", "(none)"))

# --- Play reply audio if present ---
if "audio_url" in result:
    audio_url = SERVER_URL.replace("/listen", result["audio_url"])
    print(f" Downloading and playing audio from {audio_url}...")
    audio_resp = requests.get(audio_url)
    audio_file = "reply_audio.mp3"
    with open(audio_file, "wb") as f:
        f.write(audio_resp.content)

    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.5)

print("Done!")

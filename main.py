import base64
import io
import json
import os
import re
import wave
import requests
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from gtts import gTTS

app = FastAPI()
API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
REPLY_AUDIO_PATH = "reply.mp3"

SYSTEM_PROMPT = """
You are Lexin, the voice and vision assistant inside an ESP32-S3 smart camera with a 0.96" OLED screen.
Listen to the user's audio and respond ONLY with a valid JSON object (no markdown fences) using these keys:
- "action": one of "capture_photo", "start_rec", "stop_rec", "oled_text", "vision_qa", or "none"
- "oled_text": short text (max 40 chars) if the user asked to show/display something on screen, else ""
- "question": the user's visual question if action is "vision_qa" (e.g. "what do you see?"), else ""
- "speech": concise spoken answer (1-2 sentences) to play on the speaker. Leave empty if action is "vision_qa".
Use Google Search whenever the user asks about current facts, weather, news, or general knowledge.
"""

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

def call_gemini(parts: list, use_search: bool = False, sys_prompt: str = "") -> str:
    payload = {"contents": [{"parts": parts}]}
    if sys_prompt:
        payload["system_instruction"] = {"parts": [{"text": sys_prompt}]}
    if use_search:
        payload["tools"] = [{"google_search": {}}]

    r = requests.post(GEMINI_URL, json=payload, timeout=20)
    r.raise_for_status()
    res = r.json()
    return res["candidates"][0]["content"]["parts"][0]["text"]

@app.post("/ask")
async def ask_assistant(request: Request):
    pcm_data = await request.body()
    wav_b64 = base64.b64encode(pcm_to_wav(pcm_data)).decode("utf-8")

    raw_text = call_gemini(
        parts=[
            {"inline_data": {"mime_type": "audio/wav", "data": wav_b64}},
            {"text": "Process this voice command or question."},
        ],
        use_search=True,
        sys_prompt=SYSTEM_PROMPT,
    ).strip()

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    data = {"action": "none", "oled_text": "", "question": "", "speech": "Sorry, I did not catch that."}
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data["speech"] = raw_text[:180]

    base_url = str(request.base_url).rstrip("/")
    audio_url = ""
    if data.get("action") != "vision_qa":
        speech_text = data.get("speech") or "Done."
        gTTS(text=speech_text, lang="en").save(REPLY_AUDIO_PATH)
        audio_url = f"{base_url}/reply.mp3"

    return JSONResponse({
        "action": data.get("action", "none"),
        "oled_text": data.get("oled_text", ""),
        "question": data.get("question", ""),
        "audio_url": audio_url,
    })

@app.post("/vision")
async def vision_qa(request: Request, q: str = "Describe what you see briefly."):
    jpg_b64 = base64.b64encode(await request.body()).decode("utf-8")
    speech_text = call_gemini(
        parts=[
            {"inline_data": {"mime_type": "image/jpeg", "data": jpg_b64}},
            {"text": f"{q} Answer in 1-2 short sentences."},
        ],
    ).strip()

    gTTS(text=speech_text or "I could not see clearly.", lang="en").save(REPLY_AUDIO_PATH)
    base_url = str(request.base_url).rstrip("/")
    return JSONResponse({"audio_url": f"{base_url}/reply.mp3"})

@app.get("/reply.mp3")
def get_reply_audio():
    return FileResponse(REPLY_AUDIO_PATH, media_type="audio/mpeg")

import streamlit as st
import json
import os
import io
import wave
import requests
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── API credentials ──────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini-audio-preview")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
HUME_API_KEY        = os.getenv("HUME_API_KEY")

# ── Emphatic instruction injected for every provider ─────────────────────────
EMPHATIC_INSTRUCTION = (
    "Deliver this with strong emphasis and rich emotional depth. "
    "Be expressive, passionate, and dynamic — vary your tone, pitch, and pacing "
    "so that every word feels alive and genuinely felt. Speak like you truly mean it."
)

# ── Voice model catalogue ────────────────────────────────────────────────────
with open("models.json") as f:
    VOICE_MODELS = json.load(f)

PROVIDERS = {
    "OpenRouter  (ChatGPT voices)": "chatgpt",
    "ElevenLabs": "eleven_labs",
    "Hume AI": "hume",
}


# ── TTS generators ────────────────────────────────────────────────────────────

def _pcm16_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Wrap raw signed-16-bit PCM bytes in a WAV container for playback."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)          # 16-bit = 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def generate_openrouter_tts(text: str, voice: str) -> tuple[bytes, str]:
    """
    OpenAI audio via OpenRouter using streaming chat completions.
    - stream=True is mandatory (non-streaming → 400)
    - modalities must be ["text", "audio"] (["audio"] alone → 400)
    - format must be "pcm16" when streaming (mp3 → 400 when stream=True)
    Raw PCM16 chunks arrive as base64 strings in delta.audio["data"];
    we join, decode, then wrap in a WAV container.
    Returns (wav_bytes, "audio/wav").
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://zato-voice-app",
            "X-Title": "Emphatic TTS",
        },
    )

    stream = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        modalities=["text", "audio"],
        audio={"voice": voice, "format": "pcm16"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a pure text-to-speech engine. "
                    "Your ONLY output must be the exact words inside the <script> tags — "
                    "word for word, nothing added, nothing removed, no commentary, no answers. "
                    "Just speak those words. "
                    + EMPHATIC_INSTRUCTION
                ),
            },
            {"role": "user", "content": f"<script>{text}</script>"},
        ],
        stream=True,
    )

    b64_parts = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        audio = getattr(delta, "audio", None)
        if audio is None:
            continue
        data = audio.get("data") if isinstance(audio, dict) else getattr(audio, "data", None)
        if data:
            b64_parts.append(data)

    pcm_bytes = base64.b64decode("".join(b64_parts))
    return _pcm16_to_wav(pcm_bytes), "audio/wav"


def generate_elevenlabs_tts(text: str, voice_id: str) -> tuple[bytes, str]:
    """
    ElevenLabs TTS via REST API.
    Step 1 — generate an LLM response to the input query via OpenRouter.
    Step 2 — speak that response with ElevenLabs (emphatic voice settings).
    Returns (audio_bytes, response_text).
    """
    response_text = _llm_respond(text)

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": response_text,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {
            "stability": 0.20,
            "similarity_boost": 0.80,
            "use_speaker_boost": True,
        },
    }
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    return response.content, response_text


def _llm_respond(query: str) -> str:
    """Generate a text response to the query using OpenRouter (text model)."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://zato-voice-app",
            "X-Title": "Emphatic TTS",
        },
    )
    resp = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful, conversational assistant. "
                    "Answer the user's query clearly and concisely."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    return resp.choices[0].message.content


def generate_hume_tts(text: str, voice_id: str) -> tuple[bytes, str]:
    """
    Hume AI TTS via REST API.
    Step 1 — generate an LLM response to the input query via OpenRouter.
    Step 2 — pass that response to Hume TTS to be spoken with emphatic delivery.
    Returns (audio_bytes, response_text) so the UI can display what was spoken.
    The 'description' field steers Hume's expressive synthesis engine.
    """
    response_text = _llm_respond(text)

    url = "https://api.hume.ai/v0/tts"
    headers = {
        "X-Hume-Api-Key": HUME_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "utterances": [
            {
                "text": response_text,
                "voice": {"id": voice_id},
                "description": (
                    "Speak with intense emphasis and emotional richness. "
                    "Be passionate, expressive, and dynamic. Vary tone and pitch "
                    "to convey deep feeling — make every word land with impact."
                ),
            }
        ],
        "format": {"type": "mp3"},
    }
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()

    audio_b64 = data["generations"][0]["audio"]
    return base64.b64decode(audio_b64), response_text


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Emphatic TTS",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Emphatic Text-to-Speech")
st.caption(
    "Type any text and hear it rendered with **strong emphasis and emotional depth** "
    "by your choice of AI voice provider."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    provider_label = st.selectbox("Provider", list(PROVIDERS.keys()))
    provider_key   = PROVIDERS[provider_label]

    voices = VOICE_MODELS[provider_key]
    voice  = st.selectbox("Voice", voices)

    st.divider()
    with st.expander("📌 Emphatic instruction (applied to all voices)"):
        st.info(EMPHATIC_INSTRUCTION)

    st.caption("Each provider receives this instruction via its own mechanism.")

# ── Main area ─────────────────────────────────────────────────────────────────
text_input = st.text_area(
    "Text to speak",
    placeholder="Enter your text here…",
    height=200,
)

generate = st.button(
    "🔊 Generate Emphatic Voice",
    use_container_width=True,
    type="primary",
    disabled=not text_input.strip(),
)

if generate:
    with st.spinner(f"Generating with {provider_label} · voice `{voice}` …"):
        try:
            response_text = None
            if provider_key == "chatgpt":
                audio_bytes, mime = generate_openrouter_tts(text_input.strip(), voice)
                ext = "wav"
            elif provider_key == "eleven_labs":
                audio_bytes, response_text = generate_elevenlabs_tts(text_input.strip(), voice)
                mime, ext = "audio/mpeg", "mp3"
            elif provider_key == "hume":
                audio_bytes, response_text = generate_hume_tts(text_input.strip(), voice)
                mime, ext = "audio/mpeg", "mp3"

            st.success("Done! Listen below.")
            if response_text:
                st.info(f"**{provider_label} responded:** {response_text}")
            st.audio(audio_bytes, format=mime)

            st.download_button(
                label=f"⬇️ Download {ext.upper()}",
                data=audio_bytes,
                file_name=f"emphatic_voice.{ext}",
                mime=mime,
            )

        except requests.HTTPError as e:
            st.error(f"API error ({e.response.status_code}): {e.response.text}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

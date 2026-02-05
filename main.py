#!/usr/bin/env python3
"""Record audio from microphone and transcribe using various STT providers."""

import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
import httpx
import numpy as np
import sounddevice as sd
from scipy.io import wavfile


SAMPLE_RATE = 16000
CHANNELS = 1

CONFIG_DIR = Path.home() / ".config" / "voxpaste"
CACHE_DIR = Path.home() / ".cache" / "voxpaste"

PROVIDERS = ["mistral", "openai", "groq", "deepgram"]


class Provider(Protocol):
    """Protocol for STT providers."""

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text."""
        ...


class MistralProvider:
    """Mistral Voxtral Mini provider."""

    API_URL = "https://api.mistral.ai/v1/audio/transcriptions"
    MODEL = "voxtral-mini-latest"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def transcribe(self, audio_bytes: bytes) -> str:
        response = httpx.post(
            self.API_URL,
            files={"file": ("recording.wav", audio_bytes, "audio/wav")},
            data={"model": self.MODEL},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
        )
        if response.status_code != 200:
            print(
                f"Error: API request failed with status {response.status_code}",
                file=sys.stderr,
            )
            print(f"Response: {response.text}", file=sys.stderr)
            sys.exit(1)
        return response.json().get("text", "")


class OpenAIProvider:
    """OpenAI Whisper provider."""

    API_URL = "https://api.openai.com/v1/audio/transcriptions"
    MODEL = "whisper-1"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def transcribe(self, audio_bytes: bytes) -> str:
        response = httpx.post(
            self.API_URL,
            files={"file": ("recording.wav", audio_bytes, "audio/wav")},
            data={"model": self.MODEL},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
        )
        if response.status_code != 200:
            print(
                f"Error: API request failed with status {response.status_code}",
                file=sys.stderr,
            )
            print(f"Response: {response.text}", file=sys.stderr)
            sys.exit(1)
        return response.json().get("text", "")


class GroqProvider:
    """Groq Whisper provider."""

    API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
    MODEL = "whisper-large-v3"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def transcribe(self, audio_bytes: bytes) -> str:
        response = httpx.post(
            self.API_URL,
            files={"file": ("recording.wav", audio_bytes, "audio/wav")},
            data={"model": self.MODEL},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
        )
        if response.status_code != 200:
            print(
                f"Error: API request failed with status {response.status_code}",
                file=sys.stderr,
            )
            print(f"Response: {response.text}", file=sys.stderr)
            sys.exit(1)
        return response.json().get("text", "")


class DeepgramProvider:
    """Deepgram Nova-2 provider."""

    API_URL = "https://api.deepgram.com/v1/listen?model=nova-2"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def transcribe(self, audio_bytes: bytes) -> str:
        response = httpx.post(
            self.API_URL,
            content=audio_bytes,
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav",
            },
            timeout=120.0,
        )
        if response.status_code != 200:
            print(
                f"Error: API request failed with status {response.status_code}",
                file=sys.stderr,
            )
            print(f"Response: {response.text}", file=sys.stderr)
            sys.exit(1)
        result = response.json()
        return result["results"]["channels"][0]["alternatives"][0]["transcript"]


def get_provider() -> Provider:
    """Get the configured STT provider."""
    env_file = CONFIG_DIR / ".env"
    load_dotenv(env_file)

    provider_name = os.environ.get("VOXPASTE_PROVIDER", "mistral").lower()

    print(f"Using provider: {provider_name}")

    if provider_name not in PROVIDERS:
        print(f"Error: Unknown provider '{provider_name}'", file=sys.stderr)
        print(f"Available providers: {', '.join(PROVIDERS)}", file=sys.stderr)
        sys.exit(1)

    api_key_map = {
        "mistral": "MISTRAL_API_KEY",
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "deepgram": "DEEPGRAM_API_KEY",
    }
    provider_classes = {
        "mistral": MistralProvider,
        "openai": OpenAIProvider,
        "groq": GroqProvider,
        "deepgram": DeepgramProvider,
    }

    key_name = api_key_map[provider_name]
    api_key = os.environ.get(key_name)
    if not api_key:
        print(f"Error: {key_name} not set", file=sys.stderr)
        print(
            f"Either set it in {env_file} or as an environment variable",
            file=sys.stderr,
        )
        sys.exit(1)

    return provider_classes[provider_name](api_key)


def record_audio() -> np.ndarray:
    """Record audio from microphone until Enter is pressed."""
    print("Recording... Press Enter to stop.")

    frames = []

    def callback(indata, frame_count, time_info, status):
        if status:
            print(f"Recording status: {status}", file=sys.stderr)
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.int16,
        callback=callback,
    ):
        input()

    print("Recording stopped.")

    if not frames:
        print("Error: No audio recorded", file=sys.stderr)
        sys.exit(1)

    return np.concatenate(frames, axis=0)


def audio_to_wav_bytes(audio_data: np.ndarray) -> bytes:
    """Convert numpy audio array to WAV bytes."""
    buffer = io.BytesIO()
    wavfile.write(buffer, SAMPLE_RATE, audio_data)
    buffer.seek(0)
    return buffer.read()


def copy_to_clipboard(text: str) -> None:
    """Copy text to system clipboard."""
    import platform

    clipboard_commands = []
    if platform.system() == "Darwin":
        clipboard_commands = [["pbcopy"]]
    else:
        clipboard_commands = [
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        ]

    for cmd in clipboard_commands:
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                start_new_session=True,
            )
            process.communicate(input=text.encode("utf-8"))
            if process.returncode == 0:
                print("Copied to clipboard!")
                return
        except FileNotFoundError:
            continue

    if platform.system() == "Darwin":
        print("Warning: Could not copy to clipboard", file=sys.stderr)
    else:
        print(
            "Warning: Could not copy to clipboard (install xclip or xsel)",
            file=sys.stderr,
        )


def main():
    provider = get_provider()

    audio_data = record_audio()

    duration = len(audio_data) / SAMPLE_RATE
    print(f"Recorded {duration:.1f} seconds of audio")

    audio_bytes = audio_to_wav_bytes(audio_data)

    print("Transcribing...")
    transcription = provider.transcribe(audio_bytes)

    print(f"\nTranscription:\n{transcription}")

    # Write to cache file for external clipboard access
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "last_transcription.txt").write_text(transcription)

    copy_to_clipboard(transcription)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Record audio from microphone and transcribe using various STT providers."""

import argparse
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

PROVIDERS = ["mistral", "openai", "groq", "deepgram", "openrouter"]
CLEANING_PROVIDERS = ["mistral", "openai", "groq", "openrouter"]


class Provider(Protocol):
    """Protocol for STT providers."""

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text."""
        ...


class MistralProvider:
    """Mistral Voxtral Mini provider."""

    API_URL = "https://api.mistral.ai/v1/audio/transcriptions"
    DEFAULT_MODEL = "voxtral-mini-latest"

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    def transcribe(self, audio_bytes: bytes) -> str:
        response = httpx.post(
            self.API_URL,
            files={"file": ("recording.wav", audio_bytes, "audio/wav")},
            data={"model": self.model},
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
    DEFAULT_MODEL = "whisper-1"

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    def transcribe(self, audio_bytes: bytes) -> str:
        response = httpx.post(
            self.API_URL,
            files={"file": ("recording.wav", audio_bytes, "audio/wav")},
            data={"model": self.model},
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
    DEFAULT_MODEL = "whisper-large-v3"

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    def transcribe(self, audio_bytes: bytes) -> str:
        response = httpx.post(
            self.API_URL,
            files={"file": ("recording.wav", audio_bytes, "audio/wav")},
            data={"model": self.model},
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

    API_BASE_URL = "https://api.deepgram.com/v1/listen"
    DEFAULT_MODEL = "nova-2"

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    def transcribe(self, audio_bytes: bytes) -> str:
        url = f"{self.API_BASE_URL}?model={self.model}"
        response = httpx.post(
            url,
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


class OpenRouterProvider:
    """OpenRouter provider using chat completions API with audio support."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "mistralai/voxtral-small-24b-2507"

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    def transcribe(self, audio_bytes: bytes) -> str:
        import base64

        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Prepare chat completion request with audio
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please transcribe this audio file. Only output the transcription text, nothing else.",
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_base64, "format": "wav"},
                        },
                    ],
                }
            ],
        }

        response = httpx.post(
            self.API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
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
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")


class CleaningProvider(Protocol):
    """Protocol for LLM providers that clean transcriptions."""

    def clean(self, text: str) -> str:
        """Clean the transcription text using an LLM."""
        ...


class MistralCleaningProvider:
    """Mistral chat completion provider for cleaning."""

    API_URL = "https://api.mistral.ai/v1/chat/completions"
    DEFAULT_MODEL = "mistral-small-latest"

    SYSTEM_PROMPT = """You are a text cleaning assistant. Your task is to take a voice transcription and clean it up by:
- Removing filler words (um, uh, like, you know, etc.)
- Removing repetitions
- Removing meaningless parts that don't hold information
- Fixing obvious grammatical errors from speech-to-text
- Preserving the original meaning, intent, and style of the speaker

Output ONLY the cleaned text, nothing else. Do not add explanations or comments."""

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    def clean(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        }

        response = httpx.post(
            self.API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
        if response.status_code != 200:
            print(
                f"Error: Cleaning API request failed with status {response.status_code}",
                file=sys.stderr,
            )
            print(f"Response: {response.text}", file=sys.stderr)
            sys.exit(1)

        result = response.json()
        return (
            result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        )


class OpenAICleaningProvider:
    """OpenAI chat completion provider for cleaning."""

    API_URL = "https://api.openai.com/v1/chat/completions"
    DEFAULT_MODEL = "gpt-4o-mini"

    SYSTEM_PROMPT = """You are a text cleaning assistant. Your task is to take a voice transcription and clean it up by:
- Removing filler words (um, uh, like, you know, etc.)
- Removing repetitions
- Removing meaningless parts that don't hold information
- Fixing obvious grammatical errors from speech-to-text
- Preserving the original meaning, intent, and style of the speaker

Output ONLY the cleaned text, nothing else. Do not add explanations or comments."""

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    def clean(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        }

        response = httpx.post(
            self.API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
        if response.status_code != 200:
            print(
                f"Error: Cleaning API request failed with status {response.status_code}",
                file=sys.stderr,
            )
            print(f"Response: {response.text}", file=sys.stderr)
            sys.exit(1)

        result = response.json()
        return (
            result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        )


class GroqCleaningProvider:
    """Groq chat completion provider for cleaning."""

    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    SYSTEM_PROMPT = """You are a text cleaning assistant. Your task is to take a voice transcription and clean it up by:
- Removing filler words (um, uh, like, you know, etc.)
- Removing repetitions
- Removing meaningless parts that don't hold information
- Fixing obvious grammatical errors from speech-to-text
- Preserving the original meaning, intent, and style of the speaker

Output ONLY the cleaned text, nothing else. Do not add explanations or comments."""

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    def clean(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        }

        response = httpx.post(
            self.API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
        if response.status_code != 200:
            print(
                f"Error: Cleaning API request failed with status {response.status_code}",
                file=sys.stderr,
            )
            print(f"Response: {response.text}", file=sys.stderr)
            sys.exit(1)

        result = response.json()
        return (
            result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        )


class OpenRouterCleaningProvider:
    """OpenRouter chat completion provider for cleaning."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"

    SYSTEM_PROMPT = """You are a text cleaning assistant. Your task is to take a voice transcription and clean it up by:
- Removing filler words (um, uh, like, you know, etc.)
- Removing repetitions
- Removing meaningless parts that don't hold information
- Fixing obvious grammatical errors from speech-to-text
- Preserving the original meaning, intent, and style of the speaker

Output ONLY the cleaned text, nothing else. Do not add explanations or comments."""

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    def clean(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        }

        response = httpx.post(
            self.API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
        if response.status_code != 200:
            print(
                f"Error: Cleaning API request failed with status {response.status_code}",
                file=sys.stderr,
            )
            print(f"Response: {response.text}", file=sys.stderr)
            sys.exit(1)

        result = response.json()
        return (
            result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        )


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
        "openrouter": "OPENROUTER_API_KEY",
    }
    model_env_map = {
        "mistral": "MISTRAL_MODEL",
        "openai": "OPENAI_MODEL",
        "groq": "GROQ_MODEL",
        "deepgram": "DEEPGRAM_MODEL",
        "openrouter": "OPENROUTER_MODEL",
    }
    provider_classes = {
        "mistral": MistralProvider,
        "openai": OpenAIProvider,
        "groq": GroqProvider,
        "deepgram": DeepgramProvider,
        "openrouter": OpenRouterProvider,
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

    # Get custom model if specified
    model_env_name = model_env_map[provider_name]
    model = os.environ.get(model_env_name)

    provider = provider_classes[provider_name](api_key, model)

    # Show which model is being used
    if hasattr(provider, "model"):
        print(f"Using model: {provider.model}")

    return provider


def get_cleaning_provider() -> CleaningProvider:
    """Get the configured cleaning LLM provider."""
    env_file = CONFIG_DIR / ".env"
    load_dotenv(env_file)

    # Default to the STT provider if not specified
    provider_name = os.environ.get("VOXPASTE_CLEANING_PROVIDER")
    if provider_name is None:
        provider_name = os.environ.get("VOXPASTE_PROVIDER", "mistral").lower()
    else:
        provider_name = provider_name.lower()

    print(f"Using cleaning provider: {provider_name}")

    if provider_name not in CLEANING_PROVIDERS:
        print(f"Error: Unknown cleaning provider '{provider_name}'", file=sys.stderr)
        print(
            f"Available cleaning providers: {', '.join(CLEANING_PROVIDERS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key_map = {
        "mistral": "MISTRAL_API_KEY",
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    model_env_map = {
        "mistral": "MISTRAL_CLEANING_MODEL",
        "openai": "OPENAI_CLEANING_MODEL",
        "groq": "GROQ_CLEANING_MODEL",
        "openrouter": "OPENROUTER_CLEANING_MODEL",
    }
    provider_classes = {
        "mistral": MistralCleaningProvider,
        "openai": OpenAICleaningProvider,
        "groq": GroqCleaningProvider,
        "openrouter": OpenRouterCleaningProvider,
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

    # Get custom model if specified
    model_env_name = model_env_map[provider_name]
    model = os.environ.get(model_env_name)

    cleaning_provider = provider_classes[provider_name](api_key, model)

    # Show which model is being used
    if hasattr(cleaning_provider, "model"):
        print(f"Using cleaning model: {cleaning_provider.model}")

    return cleaning_provider


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
    parser = argparse.ArgumentParser(
        description="Record audio and transcribe using various STT providers"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean the transcription using an LLM to remove filler words, repetitions, and noise",
    )

    args = parser.parse_args()

    provider = get_provider()

    audio_data = record_audio()

    duration = len(audio_data) / SAMPLE_RATE
    print(f"Recorded {duration:.1f} seconds of audio")

    audio_bytes = audio_to_wav_bytes(audio_data)

    print("Transcribing...")
    transcription = provider.transcribe(audio_bytes)

    print(f"\nTranscription:\n{transcription}")

    # Clean the transcription if requested
    if args.clean:
        print("\nCleaning transcription...")
        cleaning_provider = get_cleaning_provider()
        cleaned_text = cleaning_provider.clean(transcription)
        print(f"\nCleaned transcription:\n{cleaned_text}")
        final_text = cleaned_text
    else:
        final_text = transcription

    # Write to cache file for external clipboard access
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "last_transcription.txt").write_text(final_text)

    copy_to_clipboard(final_text)


if __name__ == "__main__":
    main()

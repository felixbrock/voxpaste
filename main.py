#!/usr/bin/env python3
"""Record audio from microphone and transcribe using various STT providers."""

import argparse
import io
import os
import platform
import subprocess
import sys
import time
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
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_API_ATTEMPTS = 3


class VoxPasteError(Exception):
    """Raised when VoxPaste encounters a user-facing error."""


class TranscriptionResult:
    """Structured result for a completed transcription."""

    def __init__(self, text: str, provider_name: str, used_fallback: bool = False):
        self.text = text
        self.provider_name = provider_name
        self.used_fallback = used_fallback


def request_with_retries(
    operation_name: str,
    request_fn,
    *,
    max_attempts: int = MAX_API_ATTEMPTS,
):
    """Execute an API request with bounded retries for transient failures."""
    last_response = None
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = request_fn()
        except httpx.RequestError as exc:
            last_exception = exc
            if attempt == max_attempts:
                break

            delay = min(2 ** (attempt - 1), 8)
            print(
                f"{operation_name} failed due to a network error "
                f"({exc.__class__.__name__}: {exc}). Retrying in {delay}s..."
            )
            time.sleep(delay)
            continue

        if response.status_code == 200:
            return response

        last_response = response
        if response.status_code not in RETRYABLE_STATUS_CODES or attempt == max_attempts:
            break

        retry_after = response.headers.get("retry-after")
        delay = int(retry_after) if retry_after and retry_after.isdigit() else min(
            2 ** (attempt - 1), 8
        )
        print(
            f"{operation_name} failed with status {response.status_code}. "
            f"Retrying in {delay}s..."
        )
        time.sleep(delay)

    if last_response is not None:
        if last_response.status_code in RETRYABLE_STATUS_CODES:
            raise VoxPasteError(
                f"{operation_name} failed after {max_attempts} attempts "
                f"(last status: {last_response.status_code})\n"
                f"Response: {last_response.text}"
            )
        raise VoxPasteError(
            f"{operation_name} failed with status {last_response.status_code}\n"
            f"Response: {last_response.text}"
        )

    if last_exception is not None:
        raise VoxPasteError(
            f"{operation_name} failed after {max_attempts} attempts\n"
            f"Network error: {last_exception}"
        )

    raise VoxPasteError(f"{operation_name} failed before receiving a response")


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
        response = request_with_retries(
            "Transcription request",
            lambda: httpx.post(
                self.API_URL,
                files={"file": ("recording.wav", audio_bytes, "audio/wav")},
                data={"model": self.model},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=120.0,
            ),
        )
        return response.json().get("text", "")


class OpenAIProvider:
    """OpenAI Whisper provider."""

    API_URL = "https://api.openai.com/v1/audio/transcriptions"
    DEFAULT_MODEL = "whisper-1"

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    def transcribe(self, audio_bytes: bytes) -> str:
        response = request_with_retries(
            "Transcription request",
            lambda: httpx.post(
                self.API_URL,
                files={"file": ("recording.wav", audio_bytes, "audio/wav")},
                data={"model": self.model},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=120.0,
            ),
        )
        return response.json().get("text", "")


class GroqProvider:
    """Groq Whisper provider."""

    API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
    DEFAULT_MODEL = "whisper-large-v3"

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    def transcribe(self, audio_bytes: bytes) -> str:
        response = request_with_retries(
            "Transcription request",
            lambda: httpx.post(
                self.API_URL,
                files={"file": ("recording.wav", audio_bytes, "audio/wav")},
                data={"model": self.model},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=120.0,
            ),
        )
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
        response = request_with_retries(
            "Transcription request",
            lambda: httpx.post(
                url,
                content=audio_bytes,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "audio/wav",
                },
                timeout=120.0,
            ),
        )
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

        response = request_with_retries(
            "Transcription request",
            lambda: httpx.post(
                self.API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            ),
        )

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

    SYSTEM_PROMPT = """You are a voice transcription cleaning assistant. Your job is to extract the essential information and intent while removing all noise.

PRESERVE:
- The exact intent and purpose (commands stay commands, questions stay questions, instructions stay instructions)
- The voice and pronouns (I, you, we, etc.)
- The core meaning and all important details

AGGRESSIVELY REMOVE:
- Filler words: um, uh, like, you know, basically, actually, I mean, sort of, kind of, essentially, obviously
- Redundant explanations and repetitive phrasing
- Meta-commentary about what you're saying ("let me explain", "what I mean is", "to clarify")
- Hedging language that adds no information ("I think", "maybe", "kind of", "sort of")
- Sentences or phrases that convey no actionable information
- False starts and immediate repetitions
- Unnecessary qualifiers and verbose phrasing

SIMPLIFY:
- Condense verbose expressions to their core meaning
- Remove tangential thoughts that don't contribute to the main point
- Keep only information-dense content

DO NOT:
- Change the fundamental structure or intent (don't turn instructions into descriptions)
- Add formality or polish that wasn't there
- Complete incomplete thoughts with your own ideas

Output ONLY the cleaned text with no explanations."""

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

        response = request_with_retries(
            "Cleaning request",
            lambda: httpx.post(
                self.API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            ),
        )

        result = response.json()
        return (
            result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        )


class OpenAICleaningProvider:
    """OpenAI chat completion provider for cleaning."""

    API_URL = "https://api.openai.com/v1/chat/completions"
    DEFAULT_MODEL = "gpt-4o-mini"

    SYSTEM_PROMPT = """You are a voice transcription cleaning assistant. Your job is to extract the essential information and intent while removing all noise.

PRESERVE:
- The exact intent and purpose (commands stay commands, questions stay questions, instructions stay instructions)
- The voice and pronouns (I, you, we, etc.)
- The core meaning and all important details

AGGRESSIVELY REMOVE:
- Filler words: um, uh, like, you know, basically, actually, I mean, sort of, kind of, essentially, obviously
- Redundant explanations and repetitive phrasing
- Meta-commentary about what you're saying ("let me explain", "what I mean is", "to clarify")
- Hedging language that adds no information ("I think", "maybe", "kind of", "sort of")
- Sentences or phrases that convey no actionable information
- False starts and immediate repetitions
- Unnecessary qualifiers and verbose phrasing

SIMPLIFY:
- Condense verbose expressions to their core meaning
- Remove tangential thoughts that don't contribute to the main point
- Keep only information-dense content

DO NOT:
- Change the fundamental structure or intent (don't turn instructions into descriptions)
- Add formality or polish that wasn't there
- Complete incomplete thoughts with your own ideas

Output ONLY the cleaned text with no explanations."""

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

        response = request_with_retries(
            "Cleaning request",
            lambda: httpx.post(
                self.API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            ),
        )

        result = response.json()
        return (
            result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        )


class GroqCleaningProvider:
    """Groq chat completion provider for cleaning."""

    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    SYSTEM_PROMPT = """You are a voice transcription cleaning assistant. Your job is to extract the essential information and intent while removing all noise.

PRESERVE:
- The exact intent and purpose (commands stay commands, questions stay questions, instructions stay instructions)
- The voice and pronouns (I, you, we, etc.)
- The core meaning and all important details

AGGRESSIVELY REMOVE:
- Filler words: um, uh, like, you know, basically, actually, I mean, sort of, kind of, essentially, obviously
- Redundant explanations and repetitive phrasing
- Meta-commentary about what you're saying ("let me explain", "what I mean is", "to clarify")
- Hedging language that adds no information ("I think", "maybe", "kind of", "sort of")
- Sentences or phrases that convey no actionable information
- False starts and immediate repetitions
- Unnecessary qualifiers and verbose phrasing

SIMPLIFY:
- Condense verbose expressions to their core meaning
- Remove tangential thoughts that don't contribute to the main point
- Keep only information-dense content

DO NOT:
- Change the fundamental structure or intent (don't turn instructions into descriptions)
- Add formality or polish that wasn't there
- Complete incomplete thoughts with your own ideas

Output ONLY the cleaned text with no explanations."""

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

        response = request_with_retries(
            "Cleaning request",
            lambda: httpx.post(
                self.API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            ),
        )

        result = response.json()
        return (
            result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        )


class OpenRouterCleaningProvider:
    """OpenRouter chat completion provider for cleaning."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"

    SYSTEM_PROMPT = """You are a voice transcription cleaning assistant. Your job is to extract the essential information and intent while removing all noise.

PRESERVE:
- The exact intent and purpose (commands stay commands, questions stay questions, instructions stay instructions)
- The voice and pronouns (I, you, we, etc.)
- The core meaning and all important details

AGGRESSIVELY REMOVE:
- Filler words: um, uh, like, you know, basically, actually, I mean, sort of, kind of, essentially, obviously
- Redundant explanations and repetitive phrasing
- Meta-commentary about what you're saying ("let me explain", "what I mean is", "to clarify")
- Hedging language that adds no information ("I think", "maybe", "kind of", "sort of")
- Sentences or phrases that convey no actionable information
- False starts and immediate repetitions
- Unnecessary qualifiers and verbose phrasing

SIMPLIFY:
- Condense verbose expressions to their core meaning
- Remove tangential thoughts that don't contribute to the main point
- Keep only information-dense content

DO NOT:
- Change the fundamental structure or intent (don't turn instructions into descriptions)
- Add formality or polish that wasn't there
- Complete incomplete thoughts with your own ideas

Output ONLY the cleaned text with no explanations."""

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

        response = request_with_retries(
            "Cleaning request",
            lambda: httpx.post(
                self.API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            ),
        )

        result = response.json()
        return (
            result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        )


def load_config() -> Path:
    """Load environment configuration and return the env file path."""
    env_file = CONFIG_DIR / ".env"
    load_dotenv(env_file)
    return env_file


def build_provider(provider_name: str, env_file: Path) -> Provider:
    """Instantiate an STT provider from environment configuration."""
    provider_name = provider_name.lower()
    if provider_name not in PROVIDERS:
        raise VoxPasteError(
            f"Unknown provider '{provider_name}'\n"
            f"Available providers: {', '.join(PROVIDERS)}"
        )

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
        raise VoxPasteError(
            f"{key_name} not set\n"
            f"Either set it in {env_file} or as an environment variable"
        )

    model_env_name = model_env_map[provider_name]
    model = os.environ.get(model_env_name)
    return provider_classes[provider_name](api_key, model)


def get_provider_config() -> tuple[str, Provider, str | None, Provider | None]:
    """Get the configured primary and optional fallback STT providers."""
    env_file = load_config()

    provider_name = os.environ.get("VOXPASTE_PROVIDER", "mistral").lower()
    provider = build_provider(provider_name, env_file)

    fallback_name = os.environ.get("VOXPASTE_FALLBACK_PROVIDER")
    fallback_provider = None
    if fallback_name is not None:
        fallback_name = fallback_name.lower()
        if fallback_name == provider_name:
            raise VoxPasteError(
                "VOXPASTE_FALLBACK_PROVIDER must differ from VOXPASTE_PROVIDER"
            )
        fallback_provider = build_provider(fallback_name, env_file)

    return provider_name, provider, fallback_name, fallback_provider


def get_cleaning_provider() -> CleaningProvider:
    """Get the configured cleaning LLM provider."""
    env_file = load_config()

    # Default to the STT provider if not specified
    provider_name = os.environ.get("VOXPASTE_CLEANING_PROVIDER")
    if provider_name is None:
        provider_name = os.environ.get("VOXPASTE_PROVIDER", "mistral").lower()
    else:
        provider_name = provider_name.lower()

    print(f"Using cleaning provider: {provider_name}")

    if provider_name not in CLEANING_PROVIDERS:
        raise VoxPasteError(
            f"Unknown cleaning provider '{provider_name}'\n"
            f"Available cleaning providers: {', '.join(CLEANING_PROVIDERS)}"
        )

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
        raise VoxPasteError(
            f"{key_name} not set\n"
            f"Either set it in {env_file} or as an environment variable"
        )

    # Get custom model if specified
    model_env_name = model_env_map[provider_name]
    model = os.environ.get(model_env_name)

    cleaning_provider = provider_classes[provider_name](api_key, model)

    # Show which model is being used
    if hasattr(cleaning_provider, "model"):
        print(f"Using cleaning model: {cleaning_provider.model}")

    return cleaning_provider


def notify_fallback_usage(primary_provider: str, fallback_provider: str) -> None:
    """Best-effort system notification when fallback transcription succeeds."""
    title = "VoxPaste used fallback transcription"
    message = (
        f"Primary provider '{primary_provider}' failed. "
        f"Transcribed with fallback provider '{fallback_provider}'."
    )

    commands: list[list[str]] = []
    system = platform.system()
    if system == "Darwin":
        safe_title = title.replace('"', '\\"')
        safe_message = message.replace('"', '\\"')
        commands = [
            [
                "osascript",
                "-e",
                f'display notification "{safe_message}" with title "{safe_title}"',
            ]
        ]
    elif system == "Linux":
        commands = [["notify-send", title, message]]

    for cmd in commands:
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    print(
        (
            "Warning: Could not send system notification about fallback provider "
            f"usage ({fallback_provider})"
        ),
        file=sys.stderr,
    )


def transcribe_with_fallback(audio_bytes: bytes) -> TranscriptionResult:
    """Transcribe audio with an optional fallback provider."""
    provider_name, provider, fallback_name, fallback_provider = get_provider_config()

    print(f"Using provider: {provider_name}")
    if hasattr(provider, "model"):
        print(f"Using model: {provider.model}")

    try:
        return TranscriptionResult(provider.transcribe(audio_bytes), provider_name)
    except VoxPasteError as primary_error:
        if fallback_provider is None or fallback_name is None:
            raise

        print(
            f"Primary provider '{provider_name}' failed: {primary_error}",
            file=sys.stderr,
        )
        print(f"Falling back to provider: {fallback_name}")
        if hasattr(fallback_provider, "model"):
            print(f"Using fallback model: {fallback_provider.model}")

        try:
            text = fallback_provider.transcribe(audio_bytes)
        except VoxPasteError as fallback_error:
            raise VoxPasteError(
                f"Primary provider '{provider_name}' failed and fallback provider "
                f"'{fallback_name}' also failed.\n\n"
                f"Primary error: {primary_error}\n\n"
                f"Fallback error: {fallback_error}"
            ) from fallback_error

        return TranscriptionResult(text, fallback_name, used_fallback=True)


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
        raise VoxPasteError("No audio recorded")

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


def maybe_wait_before_exit() -> None:
    """Keep the terminal open after an error when running interactively."""
    if not sys.stdin.isatty() or not sys.stderr.isatty():
        return

    try:
        input("\nPress Enter to close VoxPaste...")
    except EOFError:
        pass


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

    try:
        audio_data = record_audio()

        duration = len(audio_data) / SAMPLE_RATE
        print(f"Recorded {duration:.1f} seconds of audio")

        audio_bytes = audio_to_wav_bytes(audio_data)

        print("Transcribing...")
        transcription_result = transcribe_with_fallback(audio_bytes)
        transcription = transcription_result.text

        if transcription_result.used_fallback:
            notify_fallback_usage(
                os.environ.get("VOXPASTE_PROVIDER", "mistral").lower(),
                transcription_result.provider_name,
            )

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
    except VoxPasteError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        maybe_wait_before_exit()
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        maybe_wait_before_exit()
        sys.exit(1)


if __name__ == "__main__":
    main()

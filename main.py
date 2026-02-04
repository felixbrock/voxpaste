#!/usr/bin/env python3
"""Record audio from microphone and transcribe using Mistral Voxtral Mini API."""

import io
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
import httpx
import numpy as np
import sounddevice as sd
from scipy.io import wavfile


SAMPLE_RATE = 16000
CHANNELS = 1
MISTRAL_API_URL = "https://api.mistral.ai/v1/audio/transcriptions"


CONFIG_DIR = Path.home() / ".config" / "instruction-transcriber"
CACHE_DIR = Path.home() / ".cache" / "instruction-transcriber"


def get_api_key() -> str:
    """Get Mistral API key from config file or environment."""
    env_file = CONFIG_DIR / ".env"
    load_dotenv(env_file)
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print(f"Error: MISTRAL_API_KEY not set", file=sys.stderr)
        print(
            f"Either set it in {env_file} or as an environment variable",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def record_audio() -> np.ndarray:
    """Record audio from microphone until Ctrl+C is pressed."""
    print("Recording... Press Ctrl+C to stop.")

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
        try:
            while True:
                sd.sleep(100)
        except KeyboardInterrupt:
            pass

    print("\nRecording stopped.")

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


def transcribe_audio(audio_bytes: bytes, api_key: str) -> str:
    """Send audio to Mistral API for transcription."""
    print("Transcribing...")

    files = {
        "file": ("recording.wav", audio_bytes, "audio/wav"),
    }
    data = {
        "model": "voxtral-mini-latest",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    response = httpx.post(
        MISTRAL_API_URL,
        files=files,
        data=data,
        headers=headers,
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
    return result.get("text", "")


def copy_to_clipboard(text: str) -> None:
    """Copy text to system clipboard."""
    try:
        process = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE,
            start_new_session=True,
        )
        process.communicate(input=text.encode("utf-8"))
        if process.returncode != 0:
            raise subprocess.SubprocessError("xclip failed")
    except FileNotFoundError:
        try:
            process = subprocess.Popen(
                ["xsel", "--clipboard", "--input"],
                stdin=subprocess.PIPE,
                start_new_session=True,
            )
            process.communicate(input=text.encode("utf-8"))
            if process.returncode != 0:
                raise subprocess.SubprocessError("xsel failed")
        except FileNotFoundError:
            print(
                "Warning: Could not copy to clipboard (install xclip or xsel)",
                file=sys.stderr,
            )
            return

    print("Copied to clipboard!")


def main():
    api_key = get_api_key()

    audio_data = record_audio()

    duration = len(audio_data) / SAMPLE_RATE
    print(f"Recorded {duration:.1f} seconds of audio")

    audio_bytes = audio_to_wav_bytes(audio_data)

    transcription = transcribe_audio(audio_bytes, api_key)

    print(f"\nTranscription:\n{transcription}")

    # Write to cache file for external clipboard access
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "last_transcription.txt").write_text(transcription)

    copy_to_clipboard(transcription)


if __name__ == "__main__":
    main()

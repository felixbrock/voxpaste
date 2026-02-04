# Instruction Transcriber

A simple speech-to-text CLI tool that records audio from your microphone and transcribes it using the Mistral Voxtral Mini API. The transcription is automatically copied to your clipboard.

Developed and tested on Linux.

## Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A Mistral API key (get one at https://console.mistral.ai/)
- Linux system dependencies for audio recording and clipboard:

  ```bash
  # Debian/Ubuntu
  sudo apt install libportaudio2 xclip

  # Arch Linux
  sudo pacman -S portaudio xclip

  # Fedora
  sudo dnf install portaudio xclip
  ```

## Installation

### Install globally with uv (recommended)

```bash
uv tool install git+https://github.com/yourusername/instruction-transcriber.git
```

Or from a local clone:

```bash
git clone https://github.com/yourusername/instruction-transcriber.git
cd instruction-transcriber
uv tool install .
```

This makes the `transcribe` command available system-wide.

### Install globally with pipx

```bash
pipx install git+https://github.com/yourusername/instruction-transcriber.git
```

## Configuration

### Setting up your API key

The tool looks for your Mistral API key in `~/.config/instruction-transcriber/.env`.

1. Create the config directory:

   ```bash
   mkdir -p ~/.config/instruction-transcriber
   ```

2. Create the environment file with your API key:

   ```bash
   echo "MISTRAL_API_KEY=your-api-key-here" > ~/.config/instruction-transcriber/.env
   ```

3. Secure the file:
   ```bash
   chmod 600 ~/.config/instruction-transcriber/.env
   ```

Replace `your-api-key-here` with your actual Mistral API key from https://console.mistral.ai/.

## Usage

Simply run:

```bash
transcribe
```

1. The tool starts recording from your default microphone
2. Speak your instructions
3. Press `Ctrl+C` to stop recording
4. The audio is sent to Mistral for transcription
5. The transcription is printed and copied to your clipboard

The last transcription is also saved to `~/.cache/instruction-transcriber/last_transcription.txt`.

## Troubleshooting

**No audio input detected:**

- Make sure your microphone is connected and set as the default input device
- Check with `pactl list sources` to see available audio sources

**Clipboard not working:**

- Install `xclip` or `xsel` (the tool tries both)

**API key not found:**

- Verify the file exists: `cat ~/.config/instruction-transcriber/.env`
- Make sure there are no extra spaces around the `=` sign

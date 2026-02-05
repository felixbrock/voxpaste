![Voxpaste Banner](images/voxpaste_banner.png)

<p align="center">
  <strong>Your voice is the fastest interface to AI.</strong>
</p>

<p align="center">
  <a href="#installation">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Config</a>
</p>

---

**Voxpaste** is a lightweight CLI tool that turns your voice into text and drops it straight into your clipboard—ready to paste into any system that can handle textual noise and natural speech imperfections, like Claude, ChatGPT, Cursor, or other LLM-powered tools.

Stop typing long prompts. Start speaking them.

### Why Voxpaste?

- **Blazing fast** — Sub-second transcription latency
- **Multiple providers** — Choose from Mistral, OpenAI, Groq, or Deepgram
- **Zero friction** — Record → transcribe → clipboard, all in one command
- **Built for AI workflows** — Designed for developers who talk to LLMs all day. Bind it to a hotkey and invoke it from anywhere
- **Privacy-conscious** — Your audio goes directly to your chosen provider, no middlemen

## Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An API key from one of the supported providers:
  - [Mistral](https://console.mistral.ai/) (default)
  - [OpenAI](https://platform.openai.com/api-keys)
  - [Groq](https://console.groq.com/keys)
  - [Deepgram](https://console.deepgram.com/)
- System dependencies for audio recording and clipboard:

  **Linux:**

  ```bash
  # Debian/Ubuntu
  sudo apt install libportaudio2 xclip

  # Arch Linux
  sudo pacman -S portaudio xclip

  # Fedora
  sudo dnf install portaudio xclip
  ```

  **macOS:**

  ```bash
  # Install Homebrew if not already installed: https://brew.sh
  brew install portaudio
  ```

  Note: macOS has `pbcopy` built-in for clipboard support (no additional installation needed).

## Installation

### Install globally with uv (recommended)

```bash
uv tool install git+https://github.com/felixbrock/voxpaste.git
```

Or from a local clone:

```bash
git clone https://github.com/felixbrock/voxpaste.git
cd voxpaste
uv tool install .
```

This makes the `voxpaste` command available system-wide.

### Install globally with pipx

```bash
pipx install git+https://github.com/felixbrock/voxpaste.git
```

### Install with pip

```bash
pip install git+https://github.com/felixbrock/voxpaste.git
```

Or from a local clone:

```bash
git clone https://github.com/felixbrock/voxpaste.git
cd voxpaste
pip install .
```

Note: Using `uv` or `pipx` is recommended for CLI tools as they create isolated environments and avoid conflicts with system packages.

## Configuration

### Choosing a Provider

Voxpaste supports multiple speech-to-text providers. Set `VOXPASTE_PROVIDER` to choose one:

| Provider | Value               | Model            | Notes                                |
| -------- | ------------------- | ---------------- | ------------------------------------ |
| Mistral  | `mistral` (default) | Voxtral Mini     | **Best latency**, good accuracy      |
| Groq     | `groq`              | Whisper Large v3 | **Best latency**, generous free tier |
| OpenAI   | `openai`            | Whisper          | Most widely used, higher latency     |
| Deepgram | `deepgram`          | Nova-2           | Real-time focused, higher latency    |

**Recommended:** Use Mistral or Groq for the fastest transcription experience.

### Setting up your API key

Configuration is stored in `~/.config/voxpaste/.env`.

1. Create the config directory:

   ```bash
   mkdir -p ~/.config/voxpaste
   ```

2. Create the environment file with your provider and API key:

   ```bash
   # For Mistral (default)
   echo "MISTRAL_API_KEY=your-api-key-here" > ~/.config/voxpaste/.env

   # For OpenAI
   echo "VOXPASTE_PROVIDER=openai" > ~/.config/voxpaste/.env
   echo "OPENAI_API_KEY=your-api-key-here" >> ~/.config/voxpaste/.env

   # For Groq
   echo "VOXPASTE_PROVIDER=groq" > ~/.config/voxpaste/.env
   echo "GROQ_API_KEY=your-api-key-here" >> ~/.config/voxpaste/.env

   # For Deepgram
   echo "VOXPASTE_PROVIDER=deepgram" > ~/.config/voxpaste/.env
   echo "DEEPGRAM_API_KEY=your-api-key-here" >> ~/.config/voxpaste/.env
   ```

3. Secure the file:
   ```bash
   chmod 600 ~/.config/voxpaste/.env
   ```

## Usage

Simply run:

```bash
voxpaste
```

1. The tool starts recording from your default microphone
2. Speak your instructions
3. Press `Enter` to stop recording
4. The audio is sent to your configured provider for transcription
5. The transcription is printed and copied to your clipboard

The last transcription is also saved to `~/.cache/voxpaste/last_transcription.txt`.

### Pro tip: Bind to a global hotkey

For the best experience, bind `voxpaste` to a system-wide keyboard shortcut so you can trigger it from anywhere—no terminal needed.

**Linux (GNOME):**

Settings → Keyboard → Keyboard Shortcuts → Custom Shortcuts → Add:

- Name: `Voxpaste`
- Command: `voxpaste`
- Shortcut: e.g., `Super+Shift+V`

**Linux (KDE):**

System Settings → Shortcuts → Custom Shortcuts → Edit → New → Global Shortcut → Command/URL

**macOS:**

Use [Automator](https://support.apple.com/guide/automator/welcome/mac) to create a Quick Action that runs `voxpaste`, then assign a shortcut in System Settings → Keyboard → Keyboard Shortcuts → Services.

Alternatively, tools like [Raycast](https://www.raycast.com/), [Alfred](https://www.alfredapp.com/), or [Hammerspoon](https://www.hammerspoon.org/) can bind shell commands to hotkeys.

## Troubleshooting

**No audio input detected:**

- Make sure your microphone is connected and set as the default input device
- Linux: Check with `pactl list sources` to see available audio sources
- macOS: Check System Settings > Sound > Input

**Clipboard not working:**

- Linux: Install `xclip` or `xsel` (the tool tries both)
- macOS: Uses `pbcopy` which is built-in (should work automatically)

**API key not found:**

- Verify the file exists: `cat ~/.config/voxpaste/.env`
- Make sure there are no extra spaces around the `=` sign
- Ensure you have the correct API key variable for your provider (`MISTRAL_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, or `DEEPGRAM_API_KEY`)

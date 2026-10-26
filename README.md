# pi-speak

Dictate your prompts to `pi` instead of typing them.

Press **Alt+R**, speak, press **SPACE** — your words are transcribed locally using [faster-whisper](https://pypi.org/project/faster-whisper/), cleaned up (fillers gone, punctuation and question marks fixed, tech terms like `LD_LIBRARY_PATH` and "VS Code" spelled right) by the model you are currently conversing with, and sent straight into the conversation.

```
you:  "alt+r ... can you check why the build is failing on the ci pipeline? ... space"
pi:   (sees) "Can you check why the build is failing on the CI pipeline?"
```

## Setup

Requires [ffmpeg](https://ffmpeg.org/download.html) and [uv](https://docs.astral.sh/uv/getting-started/installation/).

Install it as a pi package:

```bash
pi install git:github.com/umutucak/pi-speak
cd ~/.pi/agent/git/github.com/umutucak/pi-speak && uv sync   # creates .venv
```

**If you have an NVIDIA GPU**, add the bundled CUDA runtime libs with `uv sync --extra gpu` (no system CUDA install needed).

The whisper model (~140 MB) downloads itself on first use.

Point `SPEAK_DEVICE` at your mic if it isn't the system default — an ALSA name on Linux (`aplay -l` lists them), a device index on macOS, a device name on Windows.

Works on Linux, haven't tested on macOS or Windows.

## How it works

`ffmpeg` (mic → 16 kHz mono PCM) → `stt.py` (whisper `base.en`, GPU if available, CPU otherwise) → one small LLM call with a strict "clean but don't rewrite" prompt → `pi.sendUserMessage`. No daemons, no servers, ~200 lines total.

The cleanup pass runs on **whatever model your pi session is currently using** — no extra model to install or configure. Switch models in pi and the cleanup quality switches with it.

See `AGENT.md` for the internals, gotchas, and tuning notes.

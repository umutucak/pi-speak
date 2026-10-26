# pi-speak — dictate prompts into pi

Alt+R (or `/speak`) → record mic → local whisper STT → LLM cleanup → **sent immediately** via `pi.sendUserMessage`.

## Pipeline

```
ffmpeg (mic → 16 kHz s16le mono, raw PCM to /tmp; ALSA/AVFoundation/dshow per platform)
  → stt.py (faster-whisper base.en, CUDA float16, CPU-int8 fallback)
  → ctx.modelRegistry.complete() with CLEAN_PROMPT (the session's current model)
  → pi.sendUserMessage(text)   # if editor had text, it's prepended and both are sent
```

## Non-obvious facts (not derivable from the code)

1. **python 3.14 dlopen quirk (the #1 trap):** setting `os.environ["LD_LIBRARY_PATH"]` at runtime does NOT reach `dlopen` in this environment — verified empirically. The CUDA libs therefore must be passed in the **spawn env** (`nvidiaLibPath()` in `index.ts`). `stt.py` also sets it (harmless, works on other systems). If whisper suddenly fails with `libcublas.so.12 not found`, this is why.

2. **venv layout is split:** `faster_whisper` lives in `.venv/lib64/...`, the `nvidia/*` CUDA libs in `.venv/lib/...` (both exist, both needed — only when the gpu extra is installed). That's why both dirs are scanned. Python deps are managed with **uv** (`pyproject.toml` + `uv.lock`); `uv sync` reproduces the CPU-only venv, `uv sync --extra gpu` adds the NVIDIA CUDA runtime (the `nvidia-*-cu12` wheels). The CPU int8 path is fully supported (near real-time on a modern CPU) and is the default.

3. **VRAM budget:** do not assume the GPU is free — other workloads (e.g. local LLM servers) may be occupying it. Probe the actual state with `nvidia-smi` before and after a transcription. If free VRAM is insufficient, the CUDA load fails and `stt.py` falls back to CPU int8 (slower, still fine).

4. **Mic (cross-platform via ffmpeg):** input is per-platform — Linux ALSA `default`, macOS AVFoundation `:0`, Windows dshow `Microphone`; override all with `SPEAK_DEVICE`. `ffmpeg` is killed with SIGTERM on SPACE; the file is raw PCM (no header), so no finalization needed. Requires `ffmpeg` on PATH.

5. **Cleanup model = whatever model the pi session is on** (via `ctx.modelRegistry.complete(ctx.model, ...)`). Not Ollama, not a fixed model. Quality of cleanup varies with it.

6. **`pi.sendUserMessage`** is the send path (ExtensionAPI, `dist/core/extensions/types.d.ts`). It's documented in `docs/extensions.md`. If you revert to editor-insert, use `ctx.ui.setEditorText()`.

## Tuning loop

The cleanup prompt (`CLEAN_PROMPT` in `index.ts`) is the main quality lever. It is deliberately opinionated: filler removal, technical-term repair (with concrete examples), strict punctuation (question marks for every question), no rephrasing. **When a bad transcription case appears, add it as an example in rule 2** — that's the established tuning mechanism.

## Testing without pi

```bash
cd ~/.pi/agent/extensions/pi-speak
# record 3 s of mic to raw PCM (add -f/-i args for non-Linux or non-default mics):
ffmpeg -hide_banner -loglevel error -y -t 3 -f alsa -i default -ar 16000 -ac 1 -f s16le /tmp/test.raw
# transcribe (LDP must be set in the environment, see fact #1):
LDP=$(find .venv -path "*site-packages/nvidia/*/lib" -type d | sort | tr '\n' ':')
env LD_LIBRARY_PATH="${LDP%:}" .venv/bin/python stt.py /tmp/test.raw
# check the extension loads (run from pi's package dir, e.g. `$(npm root -g)/@earendil-works/pi-coding-agent`;
# PI below = that dir — jiti aliases replace the old node_modules symlinks):
cd "$(npm root -g)/@earendil-works/pi-coding-agent"
PI="$PWD" node -e "
const PI = process.env.PI;
const { createJiti } = require('jiti');
const jiti = createJiti(process.cwd() + '/x.js', { interopDefault: true, alias: {
  '@earendil-works/pi-coding-agent': PI + '/dist/index.js',
  '@earendil-works/pi-tui': PI + '/node_modules/@earendil-works/pi-tui',
}});
const m = jiti(process.env.HOME + '/.pi/agent/extensions/pi-speak/index.ts');
m.default({registerShortcut:()=>{},registerCommand:()=>{},sendUserMessage:()=>{}});
console.log('ok');
"
```

pi-tui is not a global package — it lives inside pi-coding-agent's `node_modules/@earendil-works/`.

## Possible extensions (not done, deliberate)

- Warm whisper daemon to skip the ~2 s model load per dictation (currently loaded per use on purpose — simpler, and VRAM may be shared with other GPU workloads).
- ESC currently just cancels the recording; could also discard the LLM cleanup.
- Bigger whisper model (small.en ≈ 465 MB) if accuracy needs it and VRAM allows.

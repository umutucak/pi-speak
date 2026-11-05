# pi-speak — dictate prompts into pi

Alt+R (or `/speak`) → record mic → local whisper STT → LLM cleanup → **sent immediately** via `pi.sendUserMessage`.

## Working rules

- Tasks and follow-ups are **GitHub issues, not this file**. When something is
  worth doing, open an issue (one per thing, short) — don't accumulate TODOs
  here. This file documents how the code works, not what is pending.

## Pipeline

```
ffmpeg (mic → 16 kHz s16le mono, raw PCM to /tmp; ALSA/AVFoundation/dshow per platform)
  → whisper daemon (stt.py --serve: faster-whisper base.en, CPU int8, warm for the session)
  → ctx.modelRegistry.complete() with CLEAN_PROMPT (the session's current model)
  → pi.sendUserMessage(text)   # if editor had text, it's prepended and both are sent
```

## Non-obvious facts (not derivable from the code)

1. **Whisper runs CPU-only, as a session-scoped daemon.** The RTX 3090 is fully
occupied by local LLM serving (~23.8/24.5 GiB), so CUDA never fits — the GPU
path was dropped on 2026-11-06. `stt.py --serve` loads base.en int8 once
(~0.7 s, ~150 MB RAM); the extension spawns it on `pi.on("session_start")` and
kills it on `session_shutdown`, and the daemon also exits on stdin EOF (parent
death). Requests are file paths on stdin, exactly one text line out — safe to write while the model is still
loading (stdin buffers); stt.py prints no banner line, so every line it emits is a reply
to one request. If a dictation
finds the daemon dead, `transcribe()` lazily respawns it and falls back to a
one-shot `stt.py <file>` run on failure.

2. **python 3.14 dlopen quirk (historical):** setting `os.environ["LD_LIBRARY_PATH"]` at
runtime does NOT reach `dlopen` — this is why CUDA libs used to be passed via the spawn
env. Irrelevant now (CPU-only, no CUDA wheels), kept in case a GPU path is re-added.

3. **venv layout:** `faster_whisper` lives in `.venv/lib64/...` (a `lib/...` dir exists
too). Python deps are managed with **uv** (`pyproject.toml` + `uv.lock`); `uv sync`
reproduces the CPU-only venv. `ensureVenv()` in `index.ts` runs this automatically on
first use, so a fresh install needs no manual step.

4. **Mic (cross-platform via ffmpeg):** input is per-platform — Linux ALSA `default`, macOS AVFoundation `:0`, Windows dshow `Microphone`; override all with `SPEAK_DEVICE`. `ffmpeg` is killed with SIGTERM on SPACE; the file is raw PCM (no header), so no finalization needed. Requires `ffmpeg` on PATH.

5. **Cleanup model = whatever model the pi session is on** (via `ctx.modelRegistry.complete(ctx.model, ...)`). Not Ollama, not a fixed model. Quality of cleanup varies with it. The call passes **no reasoning option** and the session thinking level is never injected into extension `complete()` calls — reasoning state comes from pi-ai's per-API defaults, which differ per provider (see "Measured latency").

6. **`pi.sendUserMessage`** is the send path (ExtensionAPI, `dist/core/extensions/types.d.ts`). It's documented in `docs/extensions.md`. If you revert to editor-insert, use `ctx.ui.setEditorText()`.

7. **`sendUserMessage` throws while the agent is streaming unless you pass `deliverAs`** — pi's `prompt()` requires `streamingBehavior` (`'steer' | 'followUp'`) mid-run and rejects bare calls (surfaces as "Extension runtime error"). We always pass `{ deliverAs: "steer" }`: ignored when idle, steers mid-run when busy.

## Tuning loop

The cleanup prompt (`CLEAN_PROMPT` in `index.ts`) is the main quality lever. It is deliberately opinionated: filler removal, technical-term repair (with concrete examples), strict punctuation (question marks for every question), no rephrasing. **When a bad transcription case appears, add it as an example in rule 2** — that's the established tuning mechanism.

## Testing without pi

```bash
cd ~/.pi/agent/extensions/pi-speak
# record 3 s of mic to raw PCM (add -f/-i args for non-Linux or non-default mics):
ffmpeg -hide_banner -loglevel error -y -t 3 -f alsa -i default -ar 16000 -ac 1 -f s16le /tmp/test.raw
# transcribe, one-shot:
.venv/bin/python stt.py /tmp/test.raw
# or via the daemon protocol (start serve, write path, read answer):
printf '/tmp/test.raw\n' | .venv/bin/python stt.py --serve
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

## Measured latency (2026-11-05, local endpoint, RTX 3090)

Cleanup LLM step, non-streaming `/v1/chat/completions`, 170 prompt tokens
(full `CLEAN_PROMPT` + ~30-word raw dictation), request body exactly as
`ModelRegistry.complete()` produces it:

| reasoning state | Time (2 runs) | Output tokens |
|---|---|---|
| thinking OFF (`enable_thinking: false`) — what pi-speak gets | 0.54 / 0.55 s | 41 / 41 |
| thinking ON (endpoint server default) | 3.10 / 5.53 s | 364 / 657 |
| thinking ON (explicit kwarg) | 3.23 / 9.68 s | 381 / 1144 |

- The LLM step is the cheapest piece when thinking is off (~0.5 s). The whisper
  cold load (~0.7 s) is paid once per pi session — the daemon keeps the model
  warm in RAM; a warm dictation's transcription is ~0.5 s for a few seconds of
  audio (base.en int8, near real-time on the CPU).
- The API separates thinking into the message's `reasoning` field
  (`usage.completion_tokens_details.reasoning_tokens`); `content` stays clean,
  and pi-ai maps that field to a thinking block which the text filter drops —
  on this path thinking-on is pure latency, no quality impact.
- Which state a given session model lands in depends on the provider's
  per-model defaults — check your own endpoint before assuming cleanup is fast.


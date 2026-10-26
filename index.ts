/**
 * pi-speak — dictate prompts into pi.
 *
 * Alt+R (or /speak): starts recording from the mic, shows "press SPACE to stop".
 * On SPACE: stops recording, transcribes locally (faster-whisper, GPU),
 * cleans filler words with the current model, inserts text into the editor.
 *
 * Mic: system default per platform (ALSA "default" on Linux, avfoundation ":0" on macOS,
 * dshow "Microphone" on Windows); override with SPEAK_DEVICE.
 */
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { ExtensionAPI, ExtensionContext, UserMessage } from "@earendil-works/pi-coding-agent";
import { BorderedLoader } from "@earendil-works/pi-coding-agent";
import { Key } from "@earendil-works/pi-tui";

const HERE = path.dirname(new URL(import.meta.url).pathname);
const PY = path.join(HERE, ".venv", "bin", "python");
const STT = path.join(HERE, "stt.py");
// ffmpeg input args: cross-platform mic capture (Linux ALSA / macOS AVFoundation / Windows dshow)
const FFMPEG_INPUT = (() => {
  const d = process.env.SPEAK_DEVICE;
  if (process.platform === "darwin") return ["-f", "avfoundation", "-i", d ?? ":0"];
  if (process.platform === "win32") return ["-f", "dshow", "-i", `audio=${d ?? "Microphone"}`];
  return ["-f", "alsa", "-i", d ?? "default"]; // Linux
})();

// nvidia CUDA libs bundled in the venv (python3.14 dlopen won't see os.environ set at runtime)
function nvidiaLibPath(): string {
  for (const lib of ["lib", "lib64"]) {
    const base = path.join(HERE, ".venv", lib);
    const pydir = fs.existsSync(base) ? fs.readdirSync(base).find((d) => d.startsWith("python")) : undefined;
    if (!pydir) continue;
    const n = path.join(base, pydir, "site-packages", "nvidia");
    if (!fs.existsSync(n)) continue;
    return fs
      .readdirSync(n)
      .map((s) => path.join(n, s, "lib"))
      .filter((p) => fs.existsSync(p))
      .join(":");
  }
  return "";
}
const NVIDIA_LIBS = nvidiaLibPath();

const CLEAN_PROMPT = `You are the post-processor for a voice-dictation pipeline feeding a coding assistant. Input: raw speech-to-text. Output: the exact message to send.
Rules:
1. Remove fillers and false starts: um, uh, er, erm, hmm, ah, "you know", filler "like", stutters, repeated words, and abandoned fragments the speaker restarted.
2. Fix obvious speech-to-text errors: mis-heard homophones and technical terms (e.g. "V S code" -> "VSCode", "A P I" -> "API", "L D library path" -> "LD_LIBRARY_PATH"), wrong capitalization of proper nouns, programming languages, and file paths, and words split or merged by the transcriber.
3. Punctuate properly: split into sentences; statements end with periods, EVERY question ends with a question mark; commas where natural; capitalize sentence starts.
4. Keep every word the speaker actually meant, in their order. Do not rephrase, summarize, expand, or add content they did not say.
5. If it is an instruction to the assistant, keep it as a direct instruction.
Return only the final message text. No quotes, no commentary.`;

let busy = false;
let api: ExtensionAPI | undefined;

function runStt(file: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const py = spawn(PY, [STT, file], {
      stdio: ["ignore", "pipe", "pipe"],
      env: NVIDIA_LIBS
        ? { ...process.env, LD_LIBRARY_PATH: NVIDIA_LIBS + ":" + (process.env.LD_LIBRARY_PATH ?? "") }
        : process.env,
    });
    let out = "";
    let err = "";
    py.stdout.on("data", (d) => (out += d));
    py.stderr.on("data", (d) => (err += d));
    py.on("error", reject);
    py.on("close", (code) =>
      code === 0 ? resolve(out.trim()) : reject(new Error(`stt exited ${code}: ${err.slice(-300)}`)),
    );
  });
}

async function cleanWithLlm(ctx: ExtensionContext, raw: string): Promise<string> {
  const userMessage: UserMessage = {
    role: "user",
    content: [{ type: "text", text: raw }],
    timestamp: Date.now(),
  };
  const response = await ctx.modelRegistry.complete(ctx.model!, {
    systemPrompt: CLEAN_PROMPT,
    messages: [userMessage],
  });
  const text = response.content
    .filter((c): c is { type: "text"; text: string } => c.type === "text")
    .map((c) => c.text)
    .join("\n")
    .trim();
  return text || raw;
}

async function speak(ctx: ExtensionContext) {
  if (busy) {
    ctx.ui.notify("Already recording/transcribing", "info");
    return;
  }
  if (!ctx.model) {
    ctx.ui.notify("No model selected", "error");
    return;
  }
  busy = true;
  const tmp = path.join(os.tmpdir(), `speak-${Date.now()}.raw`);
  let rec: ReturnType<typeof spawn> | undefined;
  try {
    rec = spawn(
      "ffmpeg",
      ["-hide_banner", "-loglevel", "error", "-y", ...FFMPEG_INPUT, "-ar", "16000", "-ac", "1", "-f", "s16le", tmp],
      { stdio: "ignore" },
    );
    await new Promise((r) => setTimeout(r, 300));
    if (rec!.exitCode !== null) throw new Error(`mic failed to open (device ${process.env.SPEAK_DEVICE ?? "default"})`);

    // Record until SPACE (or ESC to cancel)
    await ctx.ui.custom((_tui, _theme, _kb, done) => ({
      render: () => ["  \u25cf REC \u2014 speak now, press SPACE to stop (ESC cancels)"],
      handleInput: (data: string) => {
        if (data === " " || data === "\u001b") done();
      },
      invalidate: () => {},
    }));

    rec!.kill("SIGTERM");
    await new Promise((r) => rec!.once("exit", r));

    const size = (await import("node:fs")).promises
      .stat(tmp)
      .then((s) => s.size)
      .catch(() => 0);
    if (size < 32000) {
      ctx.ui.notify("Too short \u2014 nothing recorded", "info");
      return;
    }

    // Transcribe (local) + clean (LLM)
    const text = await ctx.ui.custom((tui, theme, _kb, done) => {
      const loader = new BorderedLoader(tui, theme, `Transcribing with ${ctx.model!.id}...`);
      loader.onAbort = () => done(null);
      runStt(tmp)
        .then((raw) =>
          raw
            ? cleanWithLlm(ctx, raw)
                .then(done)
                .catch(() => done(raw))
            : done(null),
        )
        .catch(() => done(null));
      return loader;
    });

    if (!text) {
      ctx.ui.notify("No speech detected", "info");
      return;
    }
    const existing = ctx.ui.getEditorText();
    if (existing.trim()) {
      // user had text queued in the editor — send it together with the dictation
      api!.sendUserMessage(`${existing.trim()}\n${text}`);
      ctx.ui.setEditorText("");
    } else {
      api!.sendUserMessage(text);
    }
    ctx.ui.notify("Sent", "info");
  } catch (e) {
    ctx.ui.notify(`speak: ${e instanceof Error ? e.message : e}`, "error");
  } finally {
    rec?.kill("SIGKILL");
    await import("node:fs").then((m) => m.promises.unlink(tmp).catch(() => {}));
    busy = false;
  }
}

export default function (pi: ExtensionAPI) {
  api = pi;
  const describe = "Dictate: record from mic, transcribe, and send";
  pi.registerShortcut(Key.alt("r"), {
    description: describe,
    handler: (ctx) => speak(ctx),
  });
  pi.registerCommand("speak", {
    description: describe,
    handler: (_args, ctx) => speak(ctx),
  });
}

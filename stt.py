#!/usr/bin/env python3
"""Transcribe raw 16 kHz s16le mono PCM with faster-whisper (CPU, int8).

One-shot:  stt.py <file.raw>          — print transcription, exit.
Daemon:    stt.py --serve             — line protocol on stdio: read a path,
    print its transcription, repeat; exits when stdin closes (i.e. when the
    spawning pi session dies). Model is loaded once, so the ~0.9 s cold load
    is paid once per session instead of per dictation.
"""
import sys

import numpy as np
from faster_whisper import WhisperModel


def transcribe(model: WhisperModel, file: str) -> str:
    with open(file, "rb") as f:
        audio = np.frombuffer(f.read(), dtype=np.int16).astype(np.float32) / 32768.0
    segs, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
    return " ".join(s.text.strip() for s in segs).strip()


def main() -> None:
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    if "--serve" in sys.argv:
        print("ready", flush=True)
        for line in sys.stdin:  # stdin EOF (pi exit) ends the daemon
            p = line.strip()
            if not p:
                continue
            try:
                print(transcribe(model, p), flush=True)
            except Exception as e:  # keep serving; request-level failure
                print(f"ERR {e}", flush=True)
    else:
        print(transcribe(model, sys.argv[1]))


main()

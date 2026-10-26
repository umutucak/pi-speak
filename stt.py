#!/usr/bin/env python3
"""Transcribe raw 16 kHz s16le mono PCM with faster-whisper. Prints text to stdout.

Usage: stt.py <file.raw>
"""
import os
import sys

# Expose the nvidia CUDA libs bundled in this venv before ctranslate2 loads.
_here = os.path.dirname(os.path.abspath(__file__))
_ver = f"python{sys.version_info[0]}.{sys.version_info[1]}"
for _lib in ("lib", "lib64"):
    _n = os.path.join(_here, ".venv", _lib, _ver, "site-packages", "nvidia")
    if not os.path.isdir(_n):
        continue
    for _sub in sorted(os.listdir(_n)):
        _p = os.path.join(_n, _sub, "lib")
        if os.path.isdir(_p):
            os.environ["LD_LIBRARY_PATH"] = _p + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

import numpy as np
from faster_whisper import WhisperModel


def main() -> None:
    with open(sys.argv[1], "rb") as f:
        audio = np.frombuffer(f.read(), dtype=np.int16).astype(np.float32) / 32768.0
    try:
        model = WhisperModel("base.en", device="cuda", compute_type="float16")
    except Exception:
        model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
    print(" ".join(s.text.strip() for s in segs).strip())


main()

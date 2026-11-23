#!/usr/bin/env python3
"""Word-at-a-time ASS captions aligned to faster-whisper word timestamps.

The previous generator distributed words by an energy envelope, which drifted
ahead of the narration during pauses. This one uses real per-word timestamps
from Whisper and maps them onto the exact supplied script.
"""
import re
import subprocess
import sys

import numpy as np
from faster_whisper import WhisperModel

AUDIO = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "captions.ass"

TEXT = (
    "Robbing a bank in GTA 6 is the easy part. Getting away? That's where everything falls apart. "
    "Step one: your loadout. You need a full arsenal just to breach Lombank, so don't walk in light. "
    "The moment you engage security, your Wanted Level skyrockets, and the AI responds fast and aggressive. "
    "Inside, sloppy aim gets you killed. Precision shooting is mandatory, so use cover to survive the onslaught. "
    "Cash secured? Now the real challenge begins: a dynamic police chase through chaotic streets, "
    "with the heat right behind you. Lose them on the road, then switch it up. "
    "Hit the water, grab a boat, and shake off air and ground units. "
    "Every move counts in this high-stakes, realistic heist system. "
    "So... could you pull it off? Comment your plan."
)

# faster-whisper's bundled PyAV is incompatible with its own file decoder in
# this venv, so decode to raw float32 with ffmpeg and pass the array directly.
raw = subprocess.check_output(
    ["ffmpeg", "-v", "error", "-i", AUDIO, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
)
audio = np.frombuffer(raw, dtype=np.float32)

model = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=8)
segments, _ = model.transcribe(audio, language="en", word_timestamps=True, vad_filter=True)
tokens = [
    {"start": float(w.start), "end": float(w.end), "word": w.word.strip()}
    for s in segments
    for w in (s.words or [])
]

# Whisper splits "high-stakes" into two tokens; merge split hyphen words.
merged = []
for tok in tokens:
    if tok["word"].startswith("-") and merged:
        merged[-1] = {
            "start": merged[-1]["start"],
            "end": tok["end"],
            "word": merged[-1]["word"] + tok["word"],
        }
    else:
        merged.append(dict(tok))

# Map Whisper's transcription quirks to the exact supplied wording.
replacements = {
    "1.": "one:",
    "You're": "your",
    "begins.": "begins:",
    "A": "a",
    "realistic,": "realistic",
    "So,": "So...",
}
for tok in merged:
    tok["word"] = replacements.get(tok["word"], tok["word"])


def norm(s):
    return re.sub(r"[^a-z0-9']", "", s.lower())


user_words = TEXT.split()
if len(user_words) != len(merged):
    raise SystemExit(
        f"word count mismatch: script={len(user_words)} whisper={len(merged)}\n"
        + "whisper: " + " ".join(t["word"] for t in merged)
    )
mismatches = [(i, u, t["word"]) for i, (u, t) in enumerate(zip(user_words, merged)) if norm(u) != norm(t["word"])]
if mismatches:
    print("WARNING mismatches:", mismatches, file=sys.stderr)

starts = [max(0.0, t["start"]) for t in merged]
ends = []
for i in range(len(merged)):
    end = starts[i + 1] if i + 1 < len(merged) else merged[i]["end"] + 0.15
    if end <= starts[i]:  # zero-length Whisper token (e.g. duplicate "Step")
        end = starts[i] + 0.06
    ends.append(end)


def ts(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


head = """[Script Info]
ScriptType: v4.00+
PlayResX: 2160
PlayResY: 3840
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Word,DejaVu Sans,200,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,10,6,5,80,80,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
lines = [head]
for i, word in enumerate(user_words):
    display = re.sub(r"\.$|,$", "", word)
    effect = r"{\fscx118\fscy118\t(0,140,\fscx100\fscy100)}"
    lines.append(f"Dialogue: 0,{ts(starts[i])},{ts(ends[i])},Word,,0,0,0,,{effect}{display}\n")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("".join(lines))

print(f"wrote {OUT}: {len(user_words)} words, {starts[0]:.2f}-{ends[-1]:.2f}s")
for i in (0, 10, 11, 20, 60, 100, len(user_words) - 1):
    print(f"  {starts[i]:6.2f}-{ends[i]:6.2f} {user_words[i]}")

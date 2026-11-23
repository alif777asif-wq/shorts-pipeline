#!/usr/bin/env python3
"""Word-at-a-time ASS captions aligned to the narration's energy envelope."""
import re
import subprocess
import sys

AUDIO = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "captions.ass"

TEXT = ("Robbing a bank in GTA 6 is the easy part. Getting away? That's where everything falls apart. "
        "Step one: your loadout. You need a full arsenal just to breach Lombank, so don't walk in light. "
        "The moment you engage security, your Wanted Level skyrockets, and the AI responds fast and aggressive. "
        "Inside, sloppy aim gets you killed. Precision shooting is mandatory, so use cover to survive the onslaught. "
        "Cash secured? Now the real challenge begins: a dynamic police chase through chaotic streets, "
        "with the heat right behind you. Lose them on the road, then switch it up. "
        "Hit the water, grab a boat, and shake off air and ground units. "
        "Every move counts in this high-stakes, realistic heist system. "
        "So... could you pull it off? Comment your plan.")

raw = subprocess.check_output(
    ["ffmpeg", "-v", "error", "-i", AUDIO, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"])
import numpy as np
x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
hop = 320  # 20 ms
n = len(x) // hop
env = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
thr = max(0.008, env.max() * 0.06)
active = np.where(env > thr)[0]
start, end = active[0] * 0.02, (active[-1] + 1) * 0.02

words = TEXT.split()
weights = []
for w in words:
    core = re.sub(r"[,.]|(?:\.\.\.)$|^'", "", w).strip("'")
    weight = 0.6 + len(core) ** 0.75
    if w.endswith("..."):
        weight += 2.0
    elif w.endswith((".", "?", "!")):
        weight += 2.2
    elif w.endswith(","):
        weight += 0.9
    weights.append(weight)
scale = (end - start) / sum(weights)

bounds = [start]
acc = start
for w in weights:
    acc += w * scale
    bounds.append(acc)

# Snap internal word boundaries to the nearest low-energy point (±80 ms).
for i in range(1, len(bounds) - 1):
    b = bounds[i]
    lo = max(0, int((b - 0.08) / 0.02))
    hi = min(len(env) - 1, int((b + 0.08) / 0.02))
    if hi > lo:
        best = lo + int(np.argmin(env[lo:hi + 1]))
        b2 = best * 0.02
        if b2 > bounds[i - 1] + 0.1 and b2 < bounds[i + 1] - 0.1:
            bounds[i] = b2


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
for i, w in enumerate(words):
    display = re.sub(r"\.$|,$", "", w)
    effect = r"{\fscx118\fscy118\t(0,140,\fscx100\fscy100)}"
    lines.append(f"Dialogue: 0,{ts(bounds[i])},{ts(bounds[i+1])},Word,,0,0,0,,{effect}{display}\n")
open(OUT, "w", encoding="utf-8").write("".join(lines))
print(f"wrote {OUT}: {len(words)} words, {start:.2f}-{end:.2f}s")

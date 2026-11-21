#!/usr/bin/env python3
"""Shorts pipeline: fetch -> select -> cut -> compose. Raw clips stay lossless until compose."""
import argparse, json, os, pathlib, re, subprocess, sys, time
import numpy as np

W = {"gemini": 0.5, "heat": 0.3, "audio": 0.2}


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def duration(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "csv=p=0", str(path)], text=True)
    return float(out.strip())


def ts(v):
    if isinstance(v, (int, float)):
        return float(v)
    parts = [float(p) for p in str(v).strip().split(":")]
    s = 0.0
    for p in parts:
        s = s * 60 + p
    return s


def fetch(url, out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["yt-dlp", "-f", "bv*+ba/b", "--merge-output-format", "mkv", "--write-info-json",
           "-o", str(out_dir / "master.%(ext)s"), url]
    if os.environ.get("YT_COOKIES_FILE"):
        cmd[1:1] = ["--cookies", os.environ["YT_COOKIES_FILE"]]
    run(cmd)


def load_heat(info_path):
    try:
        segs = json.load(open(info_path)).get("heatmap") or []
    except (OSError, ValueError):
        return []
    return [(s["start_time"], s["end_time"], s["value"]) for s in segs]


def heat_mean(heat, a, b):
    tot = w = 0.0
    for s, e, v in heat:
        o = min(b, e) - max(a, s)
        if o > 0:
            tot += v * o
            w += o
    return tot / w if w else None


def audio_peaks(path, top=40, min_gap=20):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "8000",
                        "-f", "s16le", "-"], capture_output=True, check=True)
    x = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768
    n = len(x) // 8000
    if n < 5:
        return []
    db = 20 * np.log10(np.sqrt((x[:n * 8000].reshape(n, 8000) ** 2).mean(axis=1)) + 1e-9)
    db = np.convolve(db, np.ones(3) / 3, mode="same")
    z = (db - db.mean()) / (db.std() + 1e-9)
    peaks = []
    for t in np.argsort(-z):
        if all(abs(t - q[0]) >= min_gap for q in peaks):
            peaks.append((float(t), float(np.clip(z[t] / 4, 0, 1))))
        if len(peaks) >= top:
            break
    return peaks


def make_proxy(master, proxy):
    run(["ffmpeg", "-v", "error", "-y", "-i", str(master), "-vf", "scale=-2:360,fps=2", "-c:v", "libx264",
         "-crf", "32", "-preset", "veryfast", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(proxy)])


PROMPT = """You are an expert Shorts editor. Watch this video, including its audio, and find the {n} moments
most likely to perform as standalone Shorts of {lo}-{hi} seconds. Each must open on a strong hook, end on a
natural payoff, and never cut mid-sentence or mid-action. Return only a JSON array of objects:
{{"start": "MM:SS or HH:MM:SS", "end": "...", "score": 1-10, "reason": "...", "hook": "..."}}"""


def gemini_moments(proxy, n, lo, hi, model):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    f = client.files.upload(file=str(proxy))
    while f.state.name == "PROCESSING":
        time.sleep(5)
        f = client.files.get(name=f.name)
    if f.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini file state: {f.state.name}")
    r = client.models.generate_content(
        model=model, contents=[f, PROMPT.format(n=n, lo=lo, hi=hi)],
        config=types.GenerateContentConfig(response_mime_type="application/json"))
    return json.loads(r.text)


def rank(moments, heat, peaks, total, n, lo, hi, pad=1.0):
    scored = []
    for m in moments:
        try:
            s, e = ts(m["start"]), ts(m["end"])
        except (KeyError, ValueError):
            continue
        if not (lo - 5 <= e - s <= hi + 15) or s < 0 or e > total + 1:
            continue
        parts = {"gemini": float(m.get("score", 5)) / 10,
                 "audio": max([st for t, st in peaks if s <= t <= e], default=0.0)}
        h = heat_mean(heat, s, e)
        if h is not None:
            parts["heat"] = h
        wsum = sum(W[k] for k in parts)
        score = sum(W[k] * v for k, v in parts.items()) / wsum
        scored.append({"start": max(0.0, s - pad), "end": min(total, e + pad), "score": round(score, 3),
                       "reason": m.get("reason", ""), "hook": m.get("hook", ""),
                       "parts": {k: round(v, 3) for k, v in parts.items()}})
    picked = []
    for c in sorted(scored, key=lambda c: -c["score"]):
        if all(c["end"] <= p["start"] or c["start"] >= p["end"] for p in picked):
            picked.append(c)
        if len(picked) >= n:
            break
    return picked


def select(master, info, n, lo, hi, out_json, model):
    master = pathlib.Path(master)
    total = duration(master)
    heat, peaks = load_heat(info), audio_peaks(master)
    if os.environ.get("GEMINI_API_KEY"):
        proxy = master.with_name("proxy.mp4")
        make_proxy(master, proxy)
        moments = gemini_moments(proxy, n * 3, lo, hi, model)
    else:
        print("GEMINI_API_KEY not set: falling back to audio peaks only", file=sys.stderr)
        mid = (lo + hi) / 2
        moments = [{"start": max(0, t - mid * 0.4), "end": max(0, t - mid * 0.4) + mid, "score": 5,
                    "reason": "audio peak"} for t, _ in peaks]
    clips = rank(moments, heat, peaks, total, n, lo, hi)
    json.dump(clips, open(out_json, "w"), indent=2)
    return clips


def nearest_keyframe(master, t, window=12):
    """Return (keyframe, safe_seek): ffmpeg lands one keyframe early if asked for the exact keyframe time."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", f"{max(0, t - window)}%+{2 * window}",
         "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(master)], text=True)
    ks = sorted(float(l.split(",")[0]) for l in out.splitlines()
                if "K" in l.split(",")[-1] and l.split(",")[0] != "N/A")
    if not ks:
        return t, t
    k = min(ks, key=lambda x: abs(x - t))
    nxt = next((x for x in ks if x > k), k + 2)
    return k, k + min(0.5, (nxt - k) / 2)


def cut(master, clips_json, out_dir, exact=False):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(json.load(open(clips_json)), 1):
        out = out_dir / f"clip_{i:02d}.mkv"
        start, seek = (c["start"], c["start"]) if exact else nearest_keyframe(master, c["start"])
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{seek:.3f}", "-i", str(master),
               "-t", f"{c['end'] - seek:.3f}"]
        cmd += (["-c:v", "libx264", "-crf", "12", "-preset", "slower", "-c:a", "flac"] if exact
                else ["-c", "copy", "-avoid_negative_ts", "make_zero"])
        run(cmd + [str(out)])
        print(out)


def compose(clip, out, voice=None, subs=None, vertical=False, crf=14, preset="slower"):
    v = []
    if vertical:
        v.append("crop=ih*9/16:ih,scale=1080:1920:flags=lanczos")
    if subs:
        v.append("ass=" + str(subs).replace("\\", "/").replace(":", "\\:"))
    fc = ["[0:v]" + (",".join(v) if v else "null") + "[v]"]
    if voice:
        fc += ["[1:a]asplit=2[v1][v2]",
               "[0:a][v1]sidechaincompress=threshold=0.05:ratio=8:attack=20:release=300[g]",
               "[g][v2]amix=inputs=2:duration=first:normalize=0[m]",
               "[m]loudnorm=I=-14:TP=-1.5:LRA=11[a]"]
    else:
        fc += ["[0:a]loudnorm=I=-14:TP=-1.5:LRA=11[a]"]
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(clip)] + (["-i", str(voice)] if voice else [])
    run(cmd + ["-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
               "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k",
               "-movflags", "+faststart", str(out)])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("auto"); p.add_argument("url"); p.add_argument("--work", default="work")
    p.add_argument("--out", default="out"); p.add_argument("--n", type=int, default=5)
    p.add_argument("--min", type=int, default=20); p.add_argument("--max", type=int, default=55)
    p.add_argument("--exact", action="store_true")
    p = sub.add_parser("compose"); p.add_argument("clip"); p.add_argument("out")
    p.add_argument("--voice"); p.add_argument("--subs"); p.add_argument("--vertical", action="store_true")
    p.add_argument("--crf", type=int, default=14); p.add_argument("--preset", default="slower")
    a = ap.parse_args()
    if a.cmd == "auto":
        out = pathlib.Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        fetch(a.url, a.work)
        master = next(pathlib.Path(a.work).glob("master.mkv"))
        select(master, pathlib.Path(a.work) / "master.info.json", a.n, a.min, a.max, out / "clips.json",
               os.environ.get("GEMINI_MODEL") or "gemini-flash-latest")
        cut(master, out / "clips.json", out / "clips", a.exact)
    else:
        compose(a.clip, a.out, a.voice, a.subs, a.vertical, a.crf, a.preset)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate professional commentary script for a GTA VI clip using Gemini API video understanding.
"""

import os
import sys
import json
import time
import subprocess
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

# ─── Constants ────────────────────────────────────────────────────────────────

ANALYST_PROMPT = """You are a meticulous video analyst. Watch this clip with its audio. It is vertical gameplay from Grand Theft Auto VI. Produce a precise timeline. Return only JSON:
{"duration_s": number, "setting": "...", "beats": [{"t": seconds (one decimal), "event": "what visibly happens", "audio": "game sounds/voices/music at this moment, or 'quiet'", "intensity": 1-10}], "visual_details": ["specific, observable graphics/visual-quality details actually visible, each with approximate time, e.g. 'fur detail on the animal at 12.5s'"], "key_audio_moments": [{"t": seconds, "what": "..."}], "quiet_windows": [{"start": s, "end": e}], "payoff_time": seconds, "concerns": ["anything that looks off: not GTA VI, on-screen text, anything a narrator must avoid"]}
Use only what you can actually see and hear. Do not guess. At least 12 beats, at most about 2 per second."""

WRITER_PROMPT = """You are the best gaming Shorts commentator and scriptwriter in the business, writing for a GTA VI channel. Watch the clip again (with audio) and use the analysis JSON below. Write the voiceover for this vertical Short.

AUDIENCE AND GOAL: GTA fans just after the launch. Stop the scroll in the first two seconds, keep people to the end, make them feel how good the game looks, and make them want to comment.

VOICE: warm, genuine, excited but never shouty, like a friend who can't wait to show you something. Short spoken sentences, contractions, concrete words. No filler or cliches (no "guys", "you won't believe", "smash that like button"). No ALL CAPS, no emojis, no profanity. Written for the ear: easy to say aloud, numbers spelled out.

STRUCTURE:
- Hook, starting at 0.0: at most 9 words. A vivid, specific line taken from what is on screen (a bold observation, a question, or a tease) that makes people need the next five seconds. No fake "wait for it" promises: the payoff must really happen.
- Setup (from about 2.5 s): say what is happening in one clean sentence and give the first genuine visual praise.
- Build: one or two "notice this" moments pointing at real details in the frame (lighting, foliage, textures, animation, physics, particles, water, sound design), but only those the analysis shows are truly visible.
- Payoff: stay silent, or use at most 3 words, over the loudest or most intense game audio (gunshots, roars, impacts) so the moment lands. Then react briefly.
- Outro (last 4 seconds or so): one friendly closing thought plus a simple question that invites comments.

PRAISE RULES: celebrate the visuals and graphics quality of the newly released GTA VI in a friendly, specific way. Include at least three different observable details across the script, each tied to something actually on screen (for example fur and skin detail, lighting and atmosphere, vegetation density, animation, particle and smoke effects, depth and distance). Praise the game and the craft. Never put down other games or people. Do not claim anything you cannot see or hear in this clip: no resolution, frame rate, hardware, ray tracing, engine, development or release facts, no comparisons, no lore. If a detail is not visible, do not mention it.

TIMING: speech rate is 2.5 words per second. Each line has start and end in seconds from the clip start, with end >= start + words/2.5. No overlaps and at least 0.3 s between lines. Total 65-80 words, and total speech no more than 70% of the clip length so the game audio stays audible. Lines of at most 14 words. Anchor lines to real events from the analysis (key_audio_moments, quiet_windows, payoff_time).

Return only JSON:
{"on_screen_hook": "max 6 words for the first frame", "hook_options": ["3 alternative opening lines, each at most 9 words"], "lines": [{"start": s, "end": e, "text": "...", "purpose": "hook|setup|praise|info|reaction|outro"}], "total_words": n, "praise_points": ["each specific visual detail praised, with its time"], "notes": "anything the creator should know, including any concerns from the analysis"}"""

# ─── Configuration ────────────────────────────────────────────────────────────

CLIP_PATH = Path("/home/alif/shorts-pipeline/out/clip_01_4k_v.mp4")
PROXY_PATH = Path("/home/alif/shorts-pipeline/out/commentary/clip_01_proxy.mp4")
BEATS_JSON_PATH = Path("/home/alif/shorts-pipeline/out/commentary/clip_01_beats.json")
SCRIPT_JSON_PATH = Path("/home/alif/shorts-pipeline/out/commentary/clip_01_script.json")
SCRIPT_TXT_PATH = Path("/home/alif/shorts-pipeline/out/commentary/clip_01_script.txt")
SCRIPT_ASS_PATH = Path("/home/alif/shorts-pipeline/out/commentary/clip_01.ass")

MAX_UPLOAD_WAIT_SECONDS = 600  # 10 minutes
UPLOAD_POLL_INTERVAL = 5
MAX_API_RETRIES = 2

# ─── Helpers ──────────────────────────────────────────────────────────────────

def run_cmd(cmd: List[str], capture=True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=capture, text=True, check=False)

def get_video_duration(path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    result = run_cmd([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration", "-of", "csv=p=0", str(path)
    ])
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())

def create_proxy() -> float:
    """Create proxy video and return its duration."""
    print("Creating proxy video...")
    PROXY_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = run_cmd([
        "ffmpeg", "-y", "-i", str(CLIP_PATH),
        "-vf", "scale=720:1280",
        "-r", "30",
        "-c:v", "libx264", "-crf", "22", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "128k",
        str(PROXY_PATH)
    ])
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    duration = get_video_duration(PROXY_PATH)
    print(f"  Proxy created: {PROXY_PATH} ({duration:.2f}s)")
    return duration

def get_model_name() -> tuple[str, str]:
    """Determine which model to use with fallback."""
    # First priority: GEMINI_MODEL_PRO
    if os.environ.get("GEMINI_MODEL_PRO"):
        primary = os.environ["GEMINI_MODEL_PRO"]
    else:
        primary = "gemini-3.1-pro-preview"
    # Fallback: GEMINI_MODEL env or flash
    fallback = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    return primary, fallback

def create_client() -> genai.Client:
    """Create Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=api_key)

def upload_file(client: genai.Client, path: Path) -> types.File:
    """Upload file and poll until ACTIVE."""
    print(f"Uploading {path.name}...")
    uploaded = client.files.upload(file=str(path))
    print(f"  Uploaded: {uploaded.name} (state: {uploaded.state})")

    start_time = time.time()
    while uploaded.state != "ACTIVE":
        if uploaded.state == "FAILED":
            raise RuntimeError(f"File processing failed: {uploaded.name}")
        if time.time() - start_time > MAX_UPLOAD_WAIT_SECONDS:
            raise TimeoutError(f"File processing timed out after {MAX_UPLOAD_WAIT_SECONDS}s")
        time.sleep(UPLOAD_POLL_INTERVAL)
        uploaded = client.files.get(name=uploaded.name)
        print(f"  State: {uploaded.state}")

    print(f"  File ready: {uploaded.name}")
    return uploaded

def call_api_with_retry(client: genai.Client, model: str, contents: List, config: types.GenerateContentConfig) -> str:
    """Call Gemini API with retries for transient errors."""
    last_error = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            return response.text
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str:
                raise RuntimeError(f"Rate limit / quota error: {e}")
            if attempt < MAX_API_RETRIES:
                wait = 2 ** attempt
                print(f"  API error (attempt {attempt + 1}/{MAX_API_RETRIES + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error after {MAX_API_RETRIES + 1} attempts: {e}")
    raise last_error

def parse_json_response(text: str) -> Dict:
    """Parse JSON from API response, handling potential markdown code fences."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())

def validate_script(script: Dict, clip_duration: float, beats: Dict) -> List[str]:
    """Validate the writer's script against all rules. Returns list of violations."""
    violations = []
    lines = script.get("lines", [])

    # 1. Hook line: at most 9 words, starts at 0.3s or earlier
    hook_lines = [l for l in lines if l.get("purpose") == "hook"]
    if hook_lines:
        hook = hook_lines[0]
        hook_words = len(hook["text"].split())
        if hook_words > 9:
            violations.append(f"Hook has {hook_words} words (max 9)")
        if hook.get("start", 0) > 0.3:
            violations.append(f"Hook starts at {hook['start']}s (must be <= 0.3s)")
    else:
        violations.append("No hook line found")

    # 2. Each line: end >= start + words/2.5
    for i, line in enumerate(lines):
        words = len(line["text"].split())
        min_duration = words / 2.5
        actual_duration = line["end"] - line["start"]
        if actual_duration < min_duration - 0.01:  # small epsilon
            violations.append(f"Line {i}: duration {actual_duration:.2f}s < required {min_duration:.2f}s for {words} words")

    # 3. No overlaps, at least 0.3s between lines
    for i in range(len(lines) - 1):
        gap = lines[i + 1]["start"] - lines[i]["end"]
        if gap < 0.3 - 0.01:
            violations.append(f"Gap between line {i} and {i+1} is {gap:.2f}s (min 0.3s)")

    # 4. Total words 65-80
    total_words = script.get("total_words", sum(len(l["text"].split()) for l in lines))
    if total_words < 65 or total_words > 80:
        violations.append(f"Total words {total_words} not in range 65-80")

    # 5. Total speech time <= 70% of clip duration
    total_speech = sum(l["end"] - l["start"] for l in lines)
    if total_speech > clip_duration * 0.7 + 0.01:
        violations.append(f"Total speech {total_speech:.2f}s > 70% of clip ({clip_duration * 0.7:.2f}s)")

    # 6. Last end <= clip duration
    if lines and lines[-1]["end"] > clip_duration + 0.01:
        violations.append(f"Last line ends at {lines[-1]['end']:.2f}s > clip duration {clip_duration:.2f}s")

    # 7. At least 3 distinct praise_points
    praise_points = script.get("praise_points", [])
    if len(praise_points) < 3:
        violations.append(f"Only {len(praise_points)} praise points (need at least 3)")

    # 8. Each line at most 14 words
    for i, line in enumerate(lines):
        words = len(line["text"].split())
        if words > 14:
            violations.append(f"Line {i} has {words} words (max 14)")

    # 9. on_screen_hook max 6 words
    on_screen = script.get("on_screen_hook", "")
    if len(on_screen.split()) > 6:
        violations.append(f"on_screen_hook has {len(on_screen.split())} words (max 6)")

    # 10. hook_options: 3 alternatives, each <= 9 words
    hook_opts = script.get("hook_options", [])
    if len(hook_opts) != 3:
        violations.append(f"hook_options has {len(hook_opts)} items (need 3)")
    for i, opt in enumerate(hook_opts):
        if len(opt.split()) > 9:
            violations.append(f"hook_options[{i}] has {len(opt.split())} words (max 9)")

    return violations

def format_time_ass(seconds: float) -> str:
    """Format seconds as H:MM:SS.cs for ASS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

def write_outputs(script: Dict, clip_duration: float, beats: Dict):
    """Write all output files."""
    # JSON
    with open(SCRIPT_JSON_PATH, "w") as f:
        json.dump(script, f, indent=2)
    print(f"  Written: {SCRIPT_JSON_PATH}")

    # TXT
    with open(SCRIPT_TXT_PATH, "w") as f:
        f.write(f"On-screen hook: {script.get('on_screen_hook', '')}\n\n")
        f.write("Hook options:\n")
        for opt in script.get("hook_options", []):
            f.write(f"  - {opt}\n")
        f.write("\nTimed lines:\n")
        for line in script.get("lines", []):
            f.write(f"[{line['start']:05.1f}-{line['end']:05.1f}] {line['text']} ({line['purpose']})\n")
        f.write("\nContinuous read-through:\n")
        full_text = " ".join(l["text"] for l in script.get("lines", []))
        f.write(f"{full_text}\n\n")
        f.write(f"Word count: {script.get('total_words', 0)}\n\n")
        f.write("Praise points:\n")
        for pp in script.get("praise_points", []):
            f.write(f"  - {pp}\n")
        if script.get("notes"):
            f.write(f"\nNotes: {script['notes']}\n")
    print(f"  Written: {SCRIPT_TXT_PATH}")

    # ASS
    with open(SCRIPT_ASS_PATH, "w") as f:
        f.write("[Script Info]\n")
        f.write("Title: GTA VI Commentary\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 2160\n")
        f.write("PlayResY: 3840\n")
        f.write("WrapStyle: 0\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: Default,Sans,110,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,7,0,2,160,160,760,1\n\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        for line in script.get("lines", []):
            start = format_time_ass(line["start"])
            end = format_time_ass(line["end"])
            text = line["text"].replace("\n", "\\N")
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
    print(f"  Written: {SCRIPT_ASS_PATH}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GTA VI Commentary Script Generator")
    print("=" * 60)

    # Step 1: Create proxy and get duration
    clip_duration = create_proxy()

    # Step 2: Initialize Gemini client
    model_primary, model_fallback = get_model_name()
    print(f"Primary model: {model_primary}, Fallback: {model_fallback}")
    client = create_client()

    # Step 3: Upload proxy
    uploaded_file = upload_file(client, PROXY_PATH)

    def call_analyst(model: str) -> str:
        analyst_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )
        return call_api_with_retry(client, model, [uploaded_file, ANALYST_PROMPT], analyst_config)

    # Step 4: CALL 1 - Analyst (with fallback)
    print("\nCalling Analyst...")
    try:
        analyst_response = call_analyst(model_primary)
        used_model = model_primary
    except Exception as e:
        if "404" in str(e) or "NOT_FOUND" in str(e):
            print(f"  Primary model not found, trying fallback: {model_fallback}")
            analyst_response = call_analyst(model_fallback)
            used_model = model_fallback
        else:
            raise
    print(f"  Using model: {used_model}")
    beats = parse_json_response(analyst_response)
    with open(BEATS_JSON_PATH, "w") as f:
        json.dump(beats, f, indent=2)
    print(f"  Beats saved: {BEATS_JSON_PATH}")
    print(f"  Duration: {beats.get('duration_s', 'N/A')}s, Beats: {len(beats.get('beats', []))}")

    # Step 5: CALL 2 - Writer (use same model that worked for Analyst)
    print("\nCalling Writer...")
    writer_prompt = WRITER_PROMPT + "\n\n" + json.dumps(beats, indent=2)
    writer_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.4,
    )
    
    def call_writer(model: str) -> str:
        return call_api_with_retry(client, model, [uploaded_file, writer_prompt], writer_config)

    try:
        writer_response = call_writer(used_model)
    except Exception as e:
        if "404" in str(e) or "NOT_FOUND" in str(e):
            print(f"  Model {used_model} not found for Writer, trying fallback: {model_fallback}")
            writer_response = call_writer(model_fallback)
            used_model = model_fallback
        else:
            raise
    print(f"  Using model: {used_model}")
    script = parse_json_response(writer_response)

    # Step 6: Validate and possibly repair
    violations = validate_script(script, clip_duration, beats)
    if violations:
        print("\nValidation failed:")
        for v in violations:
            print(f"  - {v}")

        print("\nAttempting repair...")
        repair_prompt = (
            "The previous JSON had these violations:\n"
            + "\n".join(f"- {v}" for v in violations)
            + "\n\nReturn ONLY the corrected JSON with the same schema. No explanation."
        )
        repair_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
        )
        repair_response = call_api_with_retry(client, used_model, [uploaded_file, writer_prompt, script, repair_prompt], repair_config)
        script = parse_json_response(repair_response)

        # Re-validate
        violations = validate_script(script, clip_duration, beats)
        if violations:
            print("\nRepair did not fix all issues. Remaining warnings:")
            for v in violations:
                print(f"  - {v}")
        else:
            print("  Repair successful!")

    # Step 7: Write outputs
    print("\nWriting outputs...")
    write_outputs(script, clip_duration, beats)

    # Step 8: Print final script
    print("\n" + "=" * 60)
    print("FINAL SCRIPT")
    print("=" * 60)
    for line in script.get("lines", []):
        print(f"[{line['start']:05.1f}-{line['end']:05.1f}] {line['text']}")
    print(f"\nWord count: {script.get('total_words', 0)}")
    print(f"Praise points: {script.get('praise_points', [])}")
    print(f"\nOutput files:")
    print(f"  {SCRIPT_JSON_PATH}")
    print(f"  {SCRIPT_TXT_PATH}")
    print(f"  {SCRIPT_ASS_PATH}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
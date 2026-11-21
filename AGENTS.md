# Shorts pipeline: instructions for OpenCode

Run a job (nothing runs locally; compute is GitHub Actions):
    gh workflow run shorts.yml -f url="<VIDEO_URL>" -f clips=5 -f exact=false
    gh run watch            # wait for it to finish
    gh run download -n clips -D out

Results: out/clips.json (ranked moments, scores, hooks, reasons) and out/clips/clip_NN.mkv (raw, no overlays).
Add commentary, subtitles and the final encode (also off-machine, or wherever the user chooses):
    python scripts/shorts.py compose out/clips/clip_01.mkv final.mp4 --voice voice.wav --subs subs.ass [--vertical]

Rules: never re-encode raw clips before compose. Use exact=true only when a frame-accurate start matters.
Secrets: GEMINI_API_KEY (required for AI selection); YT_COOKIES (optional, Netscape cookies.txt).
Repo variable GEMINI_MODEL overrides the default model id (gemini-3.8-flash).

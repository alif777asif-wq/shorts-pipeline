# shorts-pipeline
Stage 1-3 of the Shorts workflow: fetch best source -> rank moments (Most replayed heatmap + Gemini video scan + audio peaks) -> lossless cut.
`compose` is the single final encode: ducked voiceover mix, loudnorm (-14 LUFS), burned ASS subtitles, optional 9:16.

Setup: push to a PUBLIC repo (free Actions minutes), add secret GEMINI_API_KEY, then see AGENTS.md.
Modes: exact=false -> bit-identical copy starting on a keyframe; exact=true -> frame-accurate, x264 CRF 12 + FLAC audio.
Ranking weights: Gemini 0.5, Most replayed 0.3, audio peak 0.2 (renormalized when a signal is missing).

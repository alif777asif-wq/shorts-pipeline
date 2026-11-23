#!/usr/bin/env bash
# Exact blueprint cuts, segment by segment, then lossless concat (no second video encode).
set -euo pipefail

SRC="${1:-work2/master.mkv}"
OUT="${2:-out/merged_4k_raw.mkv}"
TMP="$(mktemp -d /tmp/opencode/shorts-seg.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$(dirname "$OUT")"

ranges=("19:24" "26:30" "86:91" "225:237" "300:306" "342:351" "393:400")
: > "$TMP/list.txt"

i=0
for r in "${ranges[@]}"; do
  s=${r%:*}
  e=${r#*:}
  seg="$TMP/seg_$(printf '%02d' "$i").mkv"
  printf 'Cut %d: %s-%s (%ss)\n' "$((i+1))" "$s" "$e" "$((e-s))"
  ffmpeg -y -hide_banner -loglevel error -stats -ss "$s" -t "$((e-s))" -i "$SRC" \
    -map 0:v:0 -map 0:a:0 -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv420p \
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 -c:a flac "$seg"
  printf "file '%s'\n" "$seg" >> "$TMP/list.txt"
  i=$((i+1))
done

ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i "$TMP/list.txt" -c copy "$OUT"
printf 'Merged: %s\n' "$OUT"

#!/usr/bin/env bash
# Full-bleed 9:16 4K master. The crop window is tracked per segment using
# edge+motion energy so the busiest part of the action stays in frame.
set -euo pipefail

SRC="${1:-out/merged_4k_refined.mkv}"
OUT="${2:-out/merged_4k_refined_9x16.mkv}"

X="if(lt(t,5),1212,if(lt(t,9),1272,if(lt(t,14),1752,if(lt(t,26),1128,if(lt(t,31.75),1260,if(lt(t,40.115),1296,1524))))))"

ffmpeg -y -hide_banner -loglevel error -stats -i "$SRC" \
  -vf "crop=1215:2160:x='$X':y=0,scale=2160:3840:flags=lanczos,format=yuv420p" \
  -map 0:v:0 -map 0:a:0 -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv420p \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 -c:a copy "$OUT"

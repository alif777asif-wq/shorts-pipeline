#!/usr/bin/env bash
# Apply the refined tips to the exact-cut raw merge:
# velocity ramp on the van ditch (26-32s) and water leap (32-41s), plus a mild grade.
set -euo pipefail

SRC="${1:-out/merged_4k_raw.mkv}"
OUT="${2:-out/merged_4k_refined.mkv}"

ffmpeg -y -hide_banner -loglevel error -stats -i "$SRC" -filter_complex "
[0:v]split=6[v0i][v1i][v2i][v3i][v4i][v5i];
[v0i]trim=0:26,setpts=PTS-STARTPTS[v0];
[v1i]trim=26:29,setpts=(PTS-STARTPTS)/0.8[v1];
[v2i]trim=29:32,setpts=(PTS-STARTPTS)/1.5[v2];
[v3i]trim=32:35,setpts=(PTS-STARTPTS)/0.8[v3];
[v4i]trim=35:41,setpts=(PTS-STARTPTS)/1.3[v4];
[v5i]trim=41:48,setpts=PTS-STARTPTS[v5];
[v0][v1][v2][v3][v4][v5]concat=n=6:v=1:a=0[vcat];
[vcat]eq=saturation=1.12:contrast=1.07[v];
[0:a]asplit=6[a0i][a1i][a2i][a3i][a4i][a5i];
[a0i]atrim=0:26,asetpts=PTS-STARTPTS[a0];
[a1i]atrim=26:29,asetpts=PTS-STARTPTS,atempo=0.8[a1];
[a2i]atrim=29:32,asetpts=PTS-STARTPTS,atempo=1.5[a2];
[a3i]atrim=32:35,asetpts=PTS-STARTPTS,atempo=0.8[a3];
[a4i]atrim=35:41,asetpts=PTS-STARTPTS,atempo=1.3[a4];
[a5i]atrim=41:48,asetpts=PTS-STARTPTS[a5];
[a0][a1][a2][a3][a4][a5]concat=n=6:v=0:a=1[a]
" -map '[v]' -map '[a]' -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv420p \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
  -c:a flac -max_muxing_queue_size 4096 "$OUT"

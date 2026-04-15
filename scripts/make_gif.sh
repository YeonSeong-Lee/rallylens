#!/usr/bin/env bash
# Convert an mp4 to an optimized GIF via ffmpeg two-pass palettegen/paletteuse.
# usage: make_gif.sh <in.mp4> <out.gif> [fps=12] [width=640]
set -euo pipefail

in="${1:?usage: make_gif.sh <in.mp4> <out.gif> [fps] [width]}"
out="${2:?missing out.gif}"
fps="${3:-12}"
width="${4:-640}"

command -v ffmpeg >/dev/null || { echo "ffmpeg not on PATH (brew install ffmpeg)"; exit 1; }

palette=$(mktemp -t rallylens-palette.XXXXXX.png)
trap 'rm -f "$palette"' EXIT

ffmpeg -y -i "$in" \
    -vf "fps=${fps},scale=${width}:-1:flags=lanczos,palettegen" \
    "$palette"

ffmpeg -y -i "$in" -i "$palette" \
    -lavfi "fps=${fps},scale=${width}:-1:flags=lanczos [x]; [x][1:v] paletteuse" \
    "$out"

echo "wrote $out"

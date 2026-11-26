#!/bin/bash

cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
source ./setup_env.sh || exit 1

arch="${HAILO_ARCH:-hailo8l}"
hef="$(resolve_yolo_hef "$arch")" || exit 1
exec python3 basic_pipelines/detection_counter_screen.py \
  --input rpi \
  --arch "$arch" \
  --hef-path "$hef" \
  --use-frame --frame-rate 20 "$@"

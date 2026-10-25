#!/bin/bash
# Запуск facedetection на NPU (TIM-VX, INT8).
#
# Про LD_LIBRARY_PATH: userspace TIM-VX в /opt/tim-vx имеет версию
# 6.4.6.345497, а загруженный драйвер galcore — 6.4.8.415784. libGAL
# жёстко сравнивает версии и отказывается создавать контекст OpenVX
# ("Create tensor fail!", затем segfault). В
# /opt/tim-vx/lib-npu-patch/libGAL.so эта проверка обойдена: ветка
# сравнения версий переведена в безусловный переход на успешный путь.
# Остальные библиотеки берутся из /opt/tim-vx/lib.
#
# /dev/galcore доступен только root, поэтому скрипт сам вызывает sudo
# и сохраняет DISPLAY/XAUTHORITY, чтобы открылось окно.

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$DIR/build/src/facedetection"

if [ ! -x "$BIN" ]; then
    echo "Нет $BIN — сначала соберите проект: cmake --build build -j4" >&2
    exit 1
fi

export LD_LIBRARY_PATH="/opt/tim-vx/lib-npu-patch:/opt/tim-vx/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/pi/.Xauthority}"

if [ "$(id -u)" -ne 0 ]; then
    exec sudo env \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        DISPLAY="$DISPLAY" \
        XAUTHORITY="$XAUTHORITY" \
        "$BIN" "$@"
fi

exec "$BIN" "$@"

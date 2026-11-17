#!/usr/bin/env bash
# Vuelta final de Whisper sobre el audio de cada MP4 de C39/, a medida que el render los termina.
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")" || exit 1
for n in "$@"; do
  nn=$(printf "%02d" $n)
  until grep -q "OK .*C39_${nn}_" render_todo.log; do sleep 10; done
  "../../../TranscripcionPython/.venv/Scripts/python.exe" whisper_check.py $n final >> whisper/final.log 2>&1
done
echo FIN >> whisper/final.log

#!/usr/bin/env bash
# Render de la serie C39 (reanudable): construye las secciones pedidas (o todas las que falten) y luego el video completo.
# Uso: bash render_todo.sh [N ...]    -> log en _tmp_c39_profe/render_todo.log
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/../../_fuente" || exit 1
PY=../videos/.venv/Scripts/python.exe
LOG=../videos/_tmp_c39_profe/render_todo.log
if [ $# -gt 0 ]; then
  for n in "$@"; do echo "== seccion $n $(date +%T)" >> "$LOG"; $PY video_c39_completo.py --solo "$n" >> "$LOG" 2>&1 || echo "FALLO seccion $n" >> "$LOG"; done
fi
echo "== faltantes + completo $(date +%T)" >> "$LOG"
$PY video_c39_completo.py >> "$LOG" 2>&1 || echo "FALLO completo" >> "$LOG"
echo "== FIN $(date +%T)" >> "$LOG"

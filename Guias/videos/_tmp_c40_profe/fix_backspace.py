"""Reemplaza caracteres de retroceso (0x08) por la secuencia \\b en los archivos indicados (heredoc del Bash tool)."""
import sys
from pathlib import Path

for f in sys.argv[1:]:
    p = Path(f)
    s = p.read_text(encoding="utf-8")
    n = s.count("\x08")
    if n:
        p.write_text(s.replace("\x08", "\\b"), encoding="utf-8")
    print(p.name, "retrocesos reemplazados:", n)

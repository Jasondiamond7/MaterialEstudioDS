import sys
from pathlib import Path
from pptx import Presentation
base = next(Path(sys.argv[1]).glob("Clase 34*"))
out = []
for f in sorted(base.glob("*.pptx")):
    out.append(f"######## {f.name}")
    for i, s in enumerate(Presentation(f).slides, 1):
        out.append(f"=== slide {i}")
        for sh in s.shapes:
            if sh.has_text_frame:
                t = sh.text_frame.text.strip()
                if t: out.append(t)
            if getattr(sh, "has_table", False) and sh.has_table:
                for r in sh.table.rows:
                    out.append(" | ".join(c.text for c in r.cells))
        if s.has_notes_slide:
            n = s.notes_slide.notes_text_frame.text.strip()
            if n: out.append("[notas] " + n)
Path(sys.argv[2]).write_text("\n".join(out), encoding="utf-8")

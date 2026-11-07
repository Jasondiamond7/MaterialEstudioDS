import json, sys
from pathlib import Path
nb = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = []
for i, c in enumerate(nb["cells"]):
    out.append(f"##### [{i}] {c['cell_type']}")
    out.append("".join(c["source"]))
    for o in c.get("outputs", []):
        if "text" in o: out.append("--> " + "".join(o["text"])[:3000])
        elif "data" in o and "text/plain" in o["data"]: out.append("--> " + "".join(o["data"]["text/plain"])[:3000])
Path(sys.argv[2]).write_text("\n".join(out), encoding="utf-8")

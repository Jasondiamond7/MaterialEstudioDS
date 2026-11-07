"""Datos Club Activo (misma logica que figuras_c31.py, sin la base real que requiere openpyxl)."""
import sys
sys.path.insert(0, r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente")
__file__ = r"C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Guias\_fuente\figuras_c31.py"
src = open(__file__, encoding="utf-8").read()
cut = src.index("# ------------------------------------------------------------------ datos: base real")
exec(compile(src[:cut], "figuras_c31_head", "exec"))

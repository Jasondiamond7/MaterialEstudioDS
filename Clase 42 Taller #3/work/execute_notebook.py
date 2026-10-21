from pathlib import Path
import nbformat
from nbclient import NotebookClient

nb_path = Path(r'C:\Users\Jason Leon\Desktop\MaterialEstudioDS\Clase 42 Taller #3\Taller03_Python.ipynb')
nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(nb_path.parent)}})
client.execute()
nbformat.write(nb, nb_path)
print('executed and saved:', nb_path)
"""Convierte las fuentes ``fuentes/*.py`` (formato "percent") en notebooks ``notebooks/*.ipynb``.

Uso:
    python herramientas/construir_notebooks.py            # construye todos
    python herramientas/construir_notebooks.py 03 05      # construye solo los indicados

Si el notebook ya existe y fue ejecutado, las salidas de las celdas de código cuyo
contenido no cambió se conservan (así se puede editar el texto sin re-ejecutar).
"""

import sys  # Argumentos de la línea de comandos
from pathlib import Path  # Manejo de rutas

import nbformat  # Lectura y escritura de notebooks de Jupyter

RAIZ = Path(__file__).resolve().parents[1]  # Carpeta raíz del proyecto
FUENTES = RAIZ / "fuentes"  # Carpeta con los .py fuente
NOTEBOOKS = RAIZ / "notebooks"  # Carpeta de salida de los .ipynb


def leer_celdas(ruta):  # Separa el archivo fuente en celdas
    """Devuelve una lista de tuplas (tipo, texto)."""
    celdas, tipo, lineas = [], None, []  # Lista de celdas, tipo actual y líneas acumuladas
    for linea in ruta.read_text(encoding="utf-8").splitlines():  # Recorre el archivo línea por línea
        if linea.startswith("# %%"):  # Marcador de inicio de una nueva celda
            if tipo is not None:  # Si había una celda abierta...
                celdas.append((tipo, lineas))  # ...se guarda
            tipo = "markdown" if "[markdown]" in linea else "code"  # Tipo de la nueva celda
            lineas = []  # Reinicia las líneas acumuladas
        elif tipo is not None:  # Línea perteneciente a la celda actual
            lineas.append(linea)  # Se acumula
    if tipo is not None:  # Guarda la última celda
        celdas.append((tipo, lineas))  # Agrega la celda final
    salida = []  # Celdas limpias
    for tipo, lineas in celdas:  # Recorre las celdas crudas
        if tipo == "markdown":  # En markdown se quita el prefijo de comentario
            lineas = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lineas]  # "# texto" → "texto"
        texto = "\n".join(lineas).strip("\n")  # Une las líneas y quita saltos sobrantes
        if texto:  # Omite celdas vacías
            salida.append((tipo, texto))  # Guarda la celda
    return salida  # Devuelve la lista de celdas


def construir(ruta_fuente):  # Construye un notebook a partir de su fuente
    destino = NOTEBOOKS / (ruta_fuente.stem + ".ipynb")  # Ruta del notebook de salida
    salidas_previas = {}  # Mapa código → (salidas, contador) del notebook ya ejecutado
    if destino.exists():  # Si ya existe un notebook anterior...
        viejo = nbformat.read(destino, as_version=4)  # ...se lee
        for c in viejo.cells:  # Recorre sus celdas
            if c.cell_type == "code":  # Solo interesan las de código
                salidas_previas[c.source] = (c.outputs, c.execution_count)  # Guarda sus salidas
    nb = nbformat.v4.new_notebook()  # Notebook nuevo vacío
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}  # Kernel de Python 3
    for tipo, texto in leer_celdas(ruta_fuente):  # Recorre las celdas de la fuente
        if tipo == "markdown":  # Celda de texto
            nb.cells.append(nbformat.v4.new_markdown_cell(texto))  # Se agrega tal cual
        else:  # Celda de código
            celda = nbformat.v4.new_code_cell(texto)  # Crea la celda
            if texto in salidas_previas:  # Si el código no cambió respecto a la versión ejecutada...
                celda.outputs, celda.execution_count = salidas_previas[texto]  # ...se conservan sus salidas
            nb.cells.append(celda)  # Agrega la celda
    NOTEBOOKS.mkdir(exist_ok=True)  # Asegura que exista la carpeta de notebooks
    nbformat.write(nb, destino)  # Escribe el notebook
    print(f"Construido {destino.name} ({len(nb.cells)} celdas)")  # Informa el resultado


if __name__ == "__main__":  # Punto de entrada al ejecutar el script
    filtros = sys.argv[1:]  # Prefijos de notebooks a construir (vacío = todos)
    for fuente in sorted(FUENTES.glob("*.py")):  # Recorre las fuentes en orden
        if not filtros or any(fuente.name.startswith(f) for f in filtros):  # Aplica el filtro
            construir(fuente)  # Construye el notebook

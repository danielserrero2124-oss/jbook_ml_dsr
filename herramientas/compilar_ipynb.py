"""Une la portada (intro.md) y los 7 notebooks ejecutados en un único .ipynb para entregar.

Uso: python herramientas/compilar_ipynb.py
Salida: entrega/Proyecto_Integrador_ML_DanielSerranoRomero.ipynb
"""

import uuid  # Identificadores únicos para las celdas
from pathlib import Path  # Manejo de rutas

import nbformat  # Lectura y escritura de notebooks

RAIZ = Path(__file__).resolve().parents[1]  # Carpeta raíz del proyecto
SALIDA = RAIZ / "entrega" / "Proyecto_Integrador_ML_DanielSerranoRomero.ipynb"  # Archivo de salida
URL_LIBRO = "https://danielserrero2124-oss.github.io/jbook_ml_dsr/"  # Enlace al Jupyter Book publicado
URL_REPO = "https://github.com/danielserrero2124-oss/jbook_ml_dsr"  # Enlace al repositorio

PORTADA = f"""# Proyecto Integrador de Aprendizaje Automático
## Predicción de default en préstamos de Lending Club: scikit-learn vs. PySpark

**Autor:** Daniel Serrano Romero
**Programa:** Maestría en Matemáticas
**Asignatura:** Machine Learning

* Informe publicado (Jupyter Book): [{URL_LIBRO}]({URL_LIBRO})
* Código fuente: [{URL_REPO}]({URL_REPO})

> Este archivo reúne, en orden, los siete notebooks del proyecto **ya ejecutados** sobre el
> dataset completo, con todas sus salidas. Cada notebook se ejecutó en su propia sesión (algunos
> tardan más de una hora); para volver a ejecutarlos se recomienda usar los notebooks
> individuales del repositorio (carpeta `notebooks/`), en orden del 01 al 07."""  # Celda de portada


def main():  # Construye el notebook compilado
    libro = nbformat.v4.new_notebook()  # Notebook vacío
    libro.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}  # Kernel de Python 3
    libro.metadata["language_info"] = {"name": "python"}  # Lenguaje del notebook
    libro.cells.append(nbformat.v4.new_markdown_cell(PORTADA))  # Portada con autor y enlaces
    intro = (RAIZ / "intro.md").read_text(encoding="utf-8")  # Texto de la página de inicio del libro
    intro = intro.split("\n", 1)[1]  # Quita el título (ya está en la portada)
    libro.cells.append(nbformat.v4.new_markdown_cell(intro.strip()))  # Presentación, resultados y referencias
    for ruta in sorted((RAIZ / "notebooks").glob("0*.ipynb")):  # Recorre los notebooks 01…07 en orden
        nb = nbformat.read(ruta, as_version=4)  # Lee el notebook ejecutado
        libro.cells.append(nbformat.v4.new_markdown_cell("---"))  # Separador visual entre capítulos
        for celda in nb.cells:  # Copia cada celda con sus salidas
            celda["id"] = uuid.uuid4().hex[:8]  # Id nuevo y único dentro del archivo combinado
            libro.cells.append(celda)  # Agrega la celda
    nbformat.validate(libro)  # Verifica que el notebook resultante sea válido
    SALIDA.parent.mkdir(exist_ok=True)  # Crea la carpeta entrega/ si no existe
    nbformat.write(libro, SALIDA)  # Escribe el archivo
    print(f"{SALIDA} — {len(libro.cells)} celdas, {SALIDA.stat().st_size / 1024**2:.1f} MB")  # Resumen


if __name__ == "__main__":  # Punto de entrada
    main()  # Ejecuta la compilación

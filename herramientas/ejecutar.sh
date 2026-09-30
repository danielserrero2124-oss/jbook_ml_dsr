#!/usr/bin/env bash
# Construye y ejecuta uno o varios notebooks en orden.
# Uso: bash herramientas/ejecutar.sh 01 02 ...
set -e  # Detiene el script ante el primer error
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"  # Carpeta raíz del proyecto
PY="${PY:-$HOME/.conda/envs/ml_venv/python.exe}"  # Intérprete del entorno ml_venv (sobrescribible con PY=...)
mkdir -p "$RAIZ/outputs/logs"  # Carpeta de registros de ejecución
for n in "$@"; do  # Recorre los prefijos recibidos
  "$PY" "$RAIZ/herramientas/construir_notebooks.py" "$n"  # Construye el notebook desde su fuente
  nb=$(ls "$RAIZ/notebooks/${n}"_*.ipynb)  # Ruta del notebook construido
  echo "Ejecutando $nb"  # Informa qué se ejecuta
  (cd "$RAIZ/notebooks" && "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=-1 "$(basename "$nb")") > "$RAIZ/outputs/logs/${n}.log" 2>&1  # Ejecuta y guarda el registro
  echo "OK $nb"  # Confirma que terminó sin errores
done

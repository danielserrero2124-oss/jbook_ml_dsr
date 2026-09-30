"""Funciones auxiliares: hardware, cronómetro, JSON y figuras."""

import json  # Lectura y escritura de archivos JSON
import os  # Información del sistema operativo
import platform  # Datos de la plataforma (CPU, sistema)
import time  # Medición de tiempos con time.time()
from contextlib import contextmanager  # Permite crear bloques "with" personalizados

import psutil  # Información de memoria y núcleos físicos

from src import config as cfg  # Configuración central del proyecto


def info_hardware():  # Describe el equipo para que los tiempos sean interpretables
    """Devuelve un diccionario con CPU, núcleos y memoria RAM."""
    return {  # Diccionario con la información del equipo
        "sistema": f"{platform.system()} {platform.release()}",  # Sistema operativo y versión
        "procesador": platform.processor(),  # Descripción del procesador
        "nucleos_fisicos": psutil.cpu_count(logical=False),  # Núcleos físicos
        "nucleos_logicos": os.cpu_count(),  # Núcleos lógicos (hilos)
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),  # Memoria total en GB
        "ram_disponible_gb": round(psutil.virtual_memory().available / 1024**3, 1),  # Memoria libre en GB
        "python": platform.python_version(),  # Versión de Python
    }


@contextmanager  # Convierte la función en un administrador de contexto
def cronometro(registro, clave):  # Mide el tiempo de un bloque y lo guarda en registro[clave]
    """Uso: with cronometro(tiempos, 'entrenamiento'): ..."""
    inicio = time.time()  # Marca de tiempo inicial
    yield  # Ejecuta el bloque de código del "with"
    registro[clave] = time.time() - inicio  # Guarda los segundos transcurridos


def guardar_json(obj, ruta):  # Guarda un objeto como JSON legible
    with open(ruta, "w", encoding="utf-8") as f:  # Abre el archivo en modo escritura
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)  # Escribe con sangría y sin escapar tildes


def leer_json(ruta):  # Lee un archivo JSON
    with open(ruta, encoding="utf-8") as f:  # Abre el archivo en modo lectura
        return json.load(f)  # Devuelve su contenido como diccionario


def actualizar_json(ruta, clave, valor):  # Agrega o reemplaza una clave dentro de un JSON existente
    datos = leer_json(ruta) if os.path.exists(ruta) else {}  # Carga el archivo si existe
    datos[clave] = valor  # Actualiza la clave
    guardar_json(datos, ruta)  # Vuelve a escribir el archivo


def guardar_figura(fig, nombre):  # Exporta una figura de matplotlib a outputs/figuras
    fig.savefig(cfg.DIR_FIGURAS / f"{nombre}.png", dpi=110, bbox_inches="tight")  # Guarda en PNG

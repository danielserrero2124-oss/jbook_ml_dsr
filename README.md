# Proyecto Integrador de Aprendizaje Automático — Lending Club

Predicción de *default* (`Charged Off` = 1 vs. `Fully Paid` = 0) con seis modelos
implementados en **scikit-learn** y **PySpark**, comparación estadística (DeLong, McNemar,
bootstrap pareado con corrección de Holm) e interpretabilidad con LIME. Todo el código está
comentado línea por línea.

## Estructura

```
proyecto_integrador_ml/
├── intro.md, _config.yml, _toc.yml   # Jupyter Book
├── fuentes/        # Código fuente de cada notebook (formato "percent", fácil de leer/editar)
├── notebooks/      # Notebooks ejecutados (con salidas) que forman el libro
├── src/            # Módulos reutilizables
│   ├── config.py            # Rutas, semilla, variables, espacios de búsqueda
│   ├── preparacion.py       # Construcción de variables con pandas
│   ├── preparacion_spark.py # Lo mismo con PySpark + SparkSession + pipeline
│   ├── delong.py            # DeLong rápido (Sun y Xu, 2014) + versión directa para validar
│   ├── pruebas.py           # Métricas, umbral, McNemar, bootstrap pareado, Holm
│   ├── lime_utils.py        # Espacio de explicación de LIME compartido por los notebooks 04 y 06
│   └── utilidades.py        # Hardware, cronómetro, JSON, figuras
├── herramientas/
│   ├── construir_notebooks.py     # fuentes/*.py → notebooks/*.ipynb (conserva salidas si el código no cambió)
│   ├── ejecutar.sh                # Construye y ejecuta notebooks
│   └── ejecutar_sin_suspender.ps1 # Igual, pero impide que Windows suspenda el equipo mientras corre
├── data/           # Partición común (id, split) y matrices preprocesadas (se generan)
└── outputs/        # Métricas, puntuaciones de test, figuras, tablas (se generan)
```

## Requisitos

* Entorno `ml_venv` (Python 3.9) con `requirements.txt`.
* Java 11/17/21 para PySpark. `src/config.py` usa automáticamente el JDK de
  `C:\Program Files\Android\openjdk\jdk-21.0.8` si `JAVA_HOME` no está definido.
* El dataset `accepted_2007_to_2018Q4.csv.gz` (Kaggle, Lending Club) debe copiarse en
  `data/accepted_2007_to_2018Q4.csv.gz`; la ruta es relativa a la raíz del proyecto, así que
  funciona aunque la carpeta se mueva. No se sube al repositorio (pesa ≈ 390 MB). Otra
  ubicación: variable de entorno `LC_CSV`.

## Ejecución

En Windows conviene usar la versión que impide la suspensión del equipo (una suspensión a
mitad de la ejecución corta la comunicación interna de Spark y distorsiona los tiempos);
deja la laptop conectada al cargador y con la tapa abierta:

```powershell
powershell -ExecutionPolicy Bypass -File herramientas\ejecutar_sin_suspender.ps1 01 02 03 04 05 06 07
jupyter-book build .
```

El libro queda en `_build/html/index.html`. Tiempos aproximados en el equipo usado
(AMD Ryzen de 6 núcleos/12 hilos, 14 GB): 01 ≈ 8 min, 02 ≈ 1 min, 03 ≈ 75 min (casi todo
LinearSVC), 04 ≈ 65 min (casi todo el bosque aleatorio), 05 ≈ 5 min, 06 ≈ 1 min, 07 ≈ 25 min.

## Decisiones metodológicas principales

* **Target:** se conservan solo los préstamos con desenlace final (`Fully Paid`/`Charged Off`,
  1.345.310 filas) y luego se aplica la fórmula del enunciado. Sin muestreo.
* **Sin fuga de información:** solo variables conocidas al originar el préstamo.
* **Multicolinealidad:** se excluyen `fico_range_low` (r = 1 con `fico_range_high`),
  `installment` (r = 0.95 con `loan_amnt`) y `sub_grade` (redundante con `grade`).
* **Nulos no MCAR:** indicadores de nulo para `mths_since_last_delinq`, `emp_length` y `mort_acc`.
* **GradientBoosting:** se usa `HistGradientBoostingClassifier` porque el costo medido de
  `GradientBoostingClassifier` con el dataset completo es prohibitivo (sección 4.1).
* **Spark:** configuración del enunciado salvo `local[6]` (núcleos físicos), 6 particiones (en vez de 400) y 9g de memoria; los datos se reparticionan en 6
  particiones (el `.gz` no es divisible); Arrow desactivado por incompatibilidad con Java 21.

"""Configuración central del proyecto integrador (Lending Club).

Todos los notebooks importan este módulo para que la semilla, las rutas, las
variables y los hiperparámetros sean idénticos en scikit-learn y en PySpark.
"""

import os  # Acceso a variables de entorno del sistema operativo
import sys  # Acceso al intérprete de Python en ejecución
from pathlib import Path  # Manejo de rutas de archivos independiente del sistema operativo

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]  # Carpeta raíz del proyecto (un nivel arriba de src/)
DIR_DATOS = RAIZ / "data"  # Carpeta donde se guardan la partición y los datos preprocesados
DIR_SALIDAS = RAIZ / "outputs"  # Carpeta donde se guardan métricas, puntuaciones y figuras
DIR_FIGURAS = DIR_SALIDAS / "figuras"  # Subcarpeta para las figuras exportadas

# Ruta del CSV comprimido original, RELATIVA a la raíz del proyecto: data/accepted_2007_to_2018Q4.csv.gz
# (RAIZ se calcula a partir de la ubicación de este archivo, así que funciona aunque se mueva la carpeta).
# Puede sobrescribirse con la variable de entorno LC_CSV.
RUTA_CSV = Path(os.environ.get("LC_CSV", DIR_DATOS / "accepted_2007_to_2018Q4.csv.gz"))  # Ruta del dataset

RUTA_PARTICION = DIR_DATOS / "particion_train_test.parquet"  # Archivo con columnas id y split
RUTA_MODELADO = DIR_DATOS / "lending_club_modelado.parquet"  # Variables seleccionadas + target (para LIME)
DIR_SKLEARN_PREP = DIR_DATOS / "sklearn_prep"  # Matrices ya preprocesadas para scikit-learn
DIR_SCORES_SK = DIR_SALIDAS / "scores_sklearn"  # Puntuaciones de test de scikit-learn (una por modelo)
DIR_SCORES_SP = DIR_SALIDAS / "scores_pyspark"  # Puntuaciones de test de PySpark (una por modelo)
DIR_MODELOS_SK = DIR_SALIDAS / "modelos_sklearn"  # Mejores modelos de scikit-learn serializados
RUTA_METRICAS_SK = DIR_SALIDAS / "metricas_sklearn.json"  # Métricas y tiempos de scikit-learn
RUTA_METRICAS_SP = DIR_SALIDAS / "metricas_pyspark.json"  # Métricas y tiempos de PySpark
RUTA_TIEMPOS_PREP = DIR_SALIDAS / "tiempos_preprocesamiento.json"  # Tiempos de preprocesamiento por entorno

for _d in (DIR_DATOS, DIR_SALIDAS, DIR_FIGURAS, DIR_SKLEARN_PREP, DIR_SCORES_SK, DIR_SCORES_SP, DIR_MODELOS_SK):  # Recorre las carpetas necesarias
    _d.mkdir(parents=True, exist_ok=True)  # Crea cada carpeta si todavía no existe

# ---------------------------------------------------------------------------
# Entorno de Java / PySpark
# ---------------------------------------------------------------------------
_JDK_LOCAL = r"C:\Program Files\Android\openjdk\jdk-21.0.8"  # JDK detectado en este equipo
if "JAVA_HOME" not in os.environ and Path(_JDK_LOCAL).exists():  # Si no hay JAVA_HOME definido y el JDK existe...
    os.environ["JAVA_HOME"] = _JDK_LOCAL  # ...se usa ese JDK para arrancar la JVM de Spark
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)  # Los workers de Spark usan el mismo intérprete del notebook
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)  # El driver de Spark también usa este intérprete

# ---------------------------------------------------------------------------
# Reproducibilidad y diseño experimental
# ---------------------------------------------------------------------------
SEMILLA = 42  # Semilla fija para partición, modelos y bootstrap
PROPORCION_TEST = 0.20  # Partición 80/20 exigida por el enunciado
PLIEGUES_CV = 3  # Mismo número de pliegues en GridSearchCV y CrossValidator
B_BOOTSTRAP = 2000  # Número de remuestreos del bootstrap pareado
ALFA = 0.05  # Nivel de significancia para todas las pruebas
MARGEN_RELEVANCIA_AUC = 0.005  # Margen de relevancia práctica ΔAUC definido de antemano (sección 5.2)

# ---------------------------------------------------------------------------
# Variable objetivo
# ---------------------------------------------------------------------------
COL_ESTADO = "loan_status"  # Columna original con el estado del préstamo
COL_TARGET = "default"  # Nombre de la variable objetivo binaria
COL_ID = "id"  # Identificador único de cada préstamo
ESTADOS_FINALES = ["Fully Paid", "Charged Off"]  # Únicos estados con desenlace observado

# ---------------------------------------------------------------------------
# Variables analizadas en el EDA (todas conocidas al momento de originar el préstamo)
# ---------------------------------------------------------------------------
NUMERICAS_EDA = [  # Variables numéricas candidatas
    "loan_amnt",  # Monto solicitado
    "term_meses",  # Plazo en meses (derivado de 'term')
    "int_rate",  # Tasa de interés
    "installment",  # Cuota mensual
    "annual_inc",  # Ingreso anual declarado
    "dti",  # Razón deuda/ingreso
    "fico_range_low",  # Límite inferior del rango FICO
    "fico_range_high",  # Límite superior del rango FICO
    "emp_length_anios",  # Antigüedad laboral en años (derivada de 'emp_length')
    "delinq_2yrs",  # Morosidades en los últimos 2 años
    "inq_last_6mths",  # Consultas de crédito en los últimos 6 meses
    "open_acc",  # Líneas de crédito abiertas
    "pub_rec",  # Registros públicos negativos
    "revol_bal",  # Saldo revolvente
    "revol_util",  # Utilización del crédito revolvente
    "total_acc",  # Total de líneas de crédito históricas
    "mort_acc",  # Número de hipotecas
    "pub_rec_bankruptcies",  # Quiebras registradas
    "mths_since_last_delinq",  # Meses desde la última morosidad
    "antiguedad_credito_meses",  # Meses entre la primera línea de crédito y la emisión (derivada)
]
CATEGORICAS_EDA = [  # Variables categóricas candidatas
    "grade",  # Calificación de riesgo asignada por Lending Club
    "sub_grade",  # Sub-calificación de riesgo
    "home_ownership",  # Tipo de tenencia de vivienda
    "verification_status",  # Estado de verificación del ingreso
    "purpose",  # Propósito del préstamo
    "addr_state",  # Estado de residencia
    "initial_list_status",  # Tipo de listado inicial (w/f)
    "application_type",  # Individual o conjunta
]

# ---------------------------------------------------------------------------
# Variables finales de modelado (decididas con base en el EDA del notebook 01)
# ---------------------------------------------------------------------------
EXCLUIDAS_MULTICOLINEALIDAD = ["fico_range_low", "installment", "sub_grade"]  # Redundantes (|r| > 0.7 o V de Cramér ≈ 1)
NUMERICAS_MODELO = [c for c in NUMERICAS_EDA if c not in EXCLUIDAS_MULTICOLINEALIDAD]  # Numéricas que entran al modelo
CATEGORICAS_MODELO = [c for c in CATEGORICAS_EDA if c not in EXCLUIDAS_MULTICOLINEALIDAD]  # Categóricas que entran al modelo
VARIABLES_LOG = ["annual_inc", "revol_bal"]  # Variables muy asimétricas a las que se aplica log1p
VARIABLES_CON_INDICADOR = ["mths_since_last_delinq", "emp_length_anios", "mort_acc"]  # Variables cuyos nulos no son MCAR (ver EDA)
INDICADORES_NULO = [f"{c}_nulo" for c in VARIABLES_CON_INDICADOR]  # Nombres de los indicadores de nulo
UMBRAL_CATEGORIA_RARA = 0.01  # Categorías con frecuencia < 1 % en train se agrupan en "OTROS"
CATEGORIA_RARA = "OTROS"  # Etiqueta de la categoría agrupada
CATEGORIA_FALTANTE = "DESCONOCIDO"  # Etiqueta para valores categóricos faltantes

# ---------------------------------------------------------------------------
# Espacios de búsqueda (sección 4 del enunciado)
# ---------------------------------------------------------------------------
REG_PARAMS = [1e-6, 1e-5, 1e-4]  # regParam de PySpark para regresión logística y LinearSVC
PROFUNDIDADES_ARBOL = [5, 10, 15]  # max_depth / maxDepth del árbol de decisión
RF_ARBOLES = [10, 50, 100]  # n_estimators / numTrees del bosque aleatorio
RF_PROFUNDIDADES = [5, 10, 15]  # max_depth / maxDepth del bosque aleatorio
GB_ITERACIONES = [50, 100]  # n_estimators (max_iter) / maxIter del gradient boosting
GB_PROFUNDIDADES = [3, 5]  # max_depth / maxDepth del gradient boosting
GB_TASA_APRENDIZAJE = 0.1  # Tasa de aprendizaje fija en ambos entornos
MAX_BINS_SPARK = 32  # maxBins de los árboles de PySpark (valor por defecto, se reporta)
NUCLEOS_SPARK = 6  # Hilos de Spark = núcleos físicos del equipo (local[6]; ver notebook 04)
PARTICIONES_SPARK = NUCLEOS_SPARK  # 1 partición por hilo = 6: shuffle/paralelismo y datos (ver notebook 04)
MEMORIA_SPARK = "9g"  # Memoria del driver (= ejecutor en modo local); 8g del enunciado + 1g de margen (ver notebook 04)

NOMBRES_MODELOS = [  # Orden canónico de los seis modelos en tablas y gráficos
    "LogisticRegression",  # Regresión logística
    "DecisionTree",  # Árbol de decisión
    "RandomForest",  # Bosque aleatorio
    "GradientBoosting",  # Gradient boosting
    "LinearSVC",  # Máquina de soporte vectorial lineal
    "NaiveBayes",  # Naive Bayes gaussiano
]


def c_desde_regparam(reg_param, n):  # Convierte regParam de Spark al C de scikit-learn
    """C = 1 / (regParam × n), relación aproximada exigida por el enunciado."""
    return 1.0 / (reg_param * n)  # Devuelve el valor de C equivalente

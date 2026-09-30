"""Construcción de variables con pandas (EDA y scikit-learn).

La versión equivalente para PySpark vive en ``src/preparacion_spark.py``;
ambas aplican exactamente las mismas reglas.
"""

import numpy as np  # Cálculo numérico vectorizado
import pandas as pd  # Manipulación de datos tabulares

from src import config as cfg  # Configuración central del proyecto

# Columnas originales que hacen falta para construir todas las variables del proyecto
COLUMNAS_ORIGINALES = [
    cfg.COL_ID, cfg.COL_ESTADO, "loan_amnt", "term", "int_rate", "installment",  # Identificador, estado y condiciones del préstamo
    "annual_inc", "dti", "fico_range_low", "fico_range_high", "emp_length",  # Capacidad de pago y puntaje FICO
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",  # Historial crediticio (1)
    "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",  # Historial crediticio (2)
    "mths_since_last_delinq", "earliest_cr_line", "issue_d",  # Fechas y morosidad previa
    "grade", "sub_grade", "home_ownership", "verification_status", "purpose",  # Categóricas (1)
    "addr_state", "initial_list_status", "application_type",  # Categóricas (2)
]


def emp_length_a_numero(serie):  # Convierte textos como "10+ years" a números
    """'< 1 year' → 0, '1 year' → 1, ..., '10+ years' → 10, nulo → NaN."""
    texto = serie.astype("string")  # Asegura tipo texto (conserva los nulos como <NA>)
    numero = texto.str.extract(r"(\d+)", expand=False).astype("float")  # Extrae el primer número que aparezca
    numero = numero.where(~texto.str.contains("<", na=False), 0.0)  # "< 1 year" se interpreta como 0 años
    return numero.astype("float64")  # Devuelve la serie como flotante (NaN para faltantes)


def construir_variables(df):  # Filtra estados finales y deriva las variables del proyecto
    """Recibe el DataFrame original y devuelve id + variables de EDA + default."""
    out = df.loc[df[cfg.COL_ESTADO].isin(cfg.ESTADOS_FINALES), COLUMNAS_ORIGINALES].copy()  # Solo préstamos con desenlace observado y columnas necesarias
    out[cfg.COL_TARGET] = out[cfg.COL_ESTADO].apply(lambda x: 1 if x == "Charged Off" else 0)  # Fórmula del enunciado
    out[cfg.COL_ID] = out[cfg.COL_ID].astype("int64")  # El id pasa a entero (ya no hay filas de totales)
    out["term_meses"] = out["term"].str.extract(r"(\d+)", expand=False).astype("float64")  # " 36 months" → 36
    out["emp_length_anios"] = emp_length_a_numero(out["emp_length"])  # Antigüedad laboral numérica
    emision = pd.to_datetime(out["issue_d"], format="%b-%Y")  # Fecha de emisión del préstamo
    primera_linea = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y")  # Fecha de la primera línea de crédito
    out["antiguedad_credito_meses"] = (  # Diferencia en meses entre ambas fechas
        (emision.dt.year - primera_linea.dt.year) * 12 + (emision.dt.month - primera_linea.dt.month)  # Años*12 + meses
    ).astype("float64")  # Se guarda como flotante para admitir NaN
    out["issue_d"] = emision  # Se conserva la fecha de emisión para análisis temporales del EDA
    columnas = [cfg.COL_ID, "issue_d"] + cfg.NUMERICAS_EDA + cfg.CATEGORICAS_EDA + [cfg.COL_TARGET]  # Columnas finales
    return out[columnas].reset_index(drop=True)  # Devuelve solo las columnas necesarias con índice limpio


def categorias_frecuentes(train, columnas, umbral=cfg.UMBRAL_CATEGORIA_RARA):  # Aprende qué categorías conservar
    """Devuelve, por columna, las categorías cuya frecuencia en *train* es ≥ umbral."""
    resultado = {}  # Diccionario columna → lista de categorías frecuentes
    for col in columnas:  # Recorre cada variable categórica
        freq = train[col].fillna(cfg.CATEGORIA_FALTANTE).value_counts(normalize=True)  # Frecuencia relativa en train
        resultado[col] = sorted(freq[freq >= umbral].index.tolist())  # Guarda las categorías que superan el umbral
    return resultado  # Devuelve el diccionario aprendido


def agrupar_categorias(df, frecuentes):  # Aplica la agrupación aprendida en train
    """Rellena nulos con 'DESCONOCIDO' y agrupa categorías raras en 'OTROS'."""
    out = df.copy()  # Copia para no modificar el DataFrame original
    for col, cats in frecuentes.items():  # Recorre cada categórica con su lista de categorías frecuentes
        serie = out[col].fillna(cfg.CATEGORIA_FALTANTE)  # Sustituye los nulos por la categoría "DESCONOCIDO"
        out[col] = serie.where(serie.isin(cats), cfg.CATEGORIA_RARA)  # Lo que no es frecuente pasa a "OTROS"
    return out  # Devuelve el DataFrame transformado


def transformar_numericas(df):  # Transformaciones deterministas (no aprenden de los datos)
    """Aplica log1p a variables asimétricas y crea los indicadores de nulo."""
    out = df.copy()  # Copia para no modificar el original
    for col in cfg.VARIABLES_LOG:  # Recorre las variables muy asimétricas
        out[col] = np.log1p(out[col].clip(lower=0))  # log(1 + x), recortando negativos a 0 por seguridad
    for col, ind in zip(cfg.VARIABLES_CON_INDICADOR, cfg.INDICADORES_NULO):  # Recorre las variables con nulos no MCAR
        out[ind] = out[col].isna().astype("float64")  # 1 si el dato falta (p. ej., nunca tuvo morosidad registrada)
    return out  # Devuelve el DataFrame transformado

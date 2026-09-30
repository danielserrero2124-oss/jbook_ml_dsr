# %% [markdown]
# # 2. Partición común y preprocesamiento con scikit-learn
#
# Secciones 3.1 y 3.2 del enunciado.
#
# * La partición 80/20 estratificada se define **una sola vez** (semilla 42) y se guarda en
#   `data/particion_train_test.parquet` con las columnas `id` y `split`. PySpark leerá
#   exactamente ese archivo (no se usa `randomSplit`).
# * Todos los transformadores que aprenden parámetros (imputador, escalador, codificador y la
#   agrupación de categorías raras) se ajustan **solo con train** y luego se aplican a test.
# * El tiempo total de preprocesamiento (lectura + construcción de variables + partición +
#   transformación) se mide para compararlo con PySpark.

# %%
import sys  # Permite modificar la ruta de búsqueda de módulos
sys.path.insert(0, "..")  # Agrega la raíz del proyecto
import json  # Escritura de metadatos
import joblib  # Serialización de los transformadores ajustados
import numpy as np  # Cálculo numérico
import pandas as pd  # Manipulación de datos
from sklearn.impute import SimpleImputer  # Imputación de nulos
from sklearn.model_selection import train_test_split  # Partición estratificada
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Codificación y escalado
from src import config as cfg  # Configuración central
from src import preparacion as prep  # Construcción de variables con pandas
from src.utilidades import actualizar_json, cronometro, info_hardware  # Utilidades

tiempos = {}  # Diccionario donde se acumulan los tiempos de cada etapa
print(info_hardware())  # Hardware utilizado

# %% [markdown]
# ## Lectura y construcción de variables
#
# Se leen del CSV completo (todas las filas) solo las columnas necesarias.

# %%
with cronometro(tiempos, "lectura_csv"):  # Mide el tiempo de lectura
    df_raw = pd.read_csv(cfg.RUTA_CSV, usecols=prep.COLUMNAS_ORIGINALES, low_memory=False)  # Lee todas las filas del CSV
with cronometro(tiempos, "construccion_variables"):  # Mide el tiempo de construcción de variables
    df = prep.construir_variables(df_raw).drop(columns=["issue_d"])  # Filtra estados finales y deriva variables
del df_raw  # Libera memoria
df.to_parquet(cfg.RUTA_MODELADO, index=False)  # Guarda el dataset de trabajo en el espacio original (lo usa LIME)
print(df.shape, tiempos)  # Dimensión y tiempos parciales
df.head()  # Primeras filas

# %% [markdown]
# ## 3.1 Partición común 80/20 estratificada

# %%
with cronometro(tiempos, "particion"):  # Mide el tiempo de la partición
    ids_train, ids_test = train_test_split(  # Divide los identificadores
        df[cfg.COL_ID],  # Identificador único de cada préstamo
        test_size=cfg.PROPORCION_TEST,  # 20 % para prueba
        stratify=df[cfg.COL_TARGET],  # Estratifica por clase
        random_state=cfg.SEMILLA,  # Semilla fija
    )
    particion = pd.DataFrame({cfg.COL_ID: df[cfg.COL_ID]})  # Tabla con todos los id
    particion["split"] = np.where(particion[cfg.COL_ID].isin(set(ids_test)), "test", "train")  # Asigna train o test
    particion.to_parquet(cfg.RUTA_PARTICION, index=False)  # Guarda la asignación en Parquet
resumen = particion.merge(df[[cfg.COL_ID, cfg.COL_TARGET]], on=cfg.COL_ID).groupby("split")[cfg.COL_TARGET].agg(["count", "mean"])  # Tamaño y tasa de default por conjunto
resumen.columns = ["n", "tasa_default"]  # Nombres descriptivos
resumen  # Muestra que la estratificación conserva la proporción de clases

# %%
es_test = df[cfg.COL_ID].isin(set(ids_test)).to_numpy()  # Máscara de filas de test
train = df.loc[~es_test].reset_index(drop=True)  # Conjunto de entrenamiento
test = df.loc[es_test].reset_index(drop=True)  # Conjunto de prueba
print(f"train: {len(train):,}  test: {len(test):,}")  # Tamaños

# %% [markdown]
# ## 3.2 Preprocesamiento (ajustado solo con train)
#
# 1. Transformaciones deterministas: `log1p` en `annual_inc` y `revol_bal` (asimetría muy alta)
#    e indicadores de nulo de `mths_since_last_delinq`, `emp_length_anios` y `mort_acc` (nulos no MCAR, ver EDA).
# 2. Categóricas: nulos → `DESCONOCIDO`; categorías con < 1 % en train → `OTROS`; `OneHotEncoder`.
# 3. Numéricas: imputación con la mediana de train y `StandardScaler` (necesario para la
#    regresión logística y LinearSVC; los árboles son invariantes a esta transformación).
#
# Se excluyen `fico_range_low`, `installment` y `sub_grade` por multicolinealidad (EDA).

# %%
with cronometro(tiempos, "ajuste_y_transformacion"):  # Mide el tiempo de ajuste y transformación
    frecuentes = prep.categorias_frecuentes(train, cfg.CATEGORICAS_MODELO)  # Categorías frecuentes aprendidas en train
    train_t = prep.transformar_numericas(prep.agrupar_categorias(train, frecuentes))  # Aplica reglas a train
    test_t = prep.transformar_numericas(prep.agrupar_categorias(test, frecuentes))  # Aplica las mismas reglas a test
    numericas = cfg.NUMERICAS_MODELO + cfg.INDICADORES_NULO  # Columnas numéricas finales (incluye indicadores de nulo)
    imputador = SimpleImputer(strategy="median").fit(train_t[numericas])  # Aprende la mediana de cada numérica en train
    escalador = StandardScaler().fit(imputador.transform(train_t[numericas]))  # Aprende media y desviación en train
    codificador = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32).fit(train_t[cfg.CATEGORICAS_MODELO])  # Aprende las categorías en train

    def transformar(d):  # Aplica los transformadores ajustados a un conjunto
        num = escalador.transform(imputador.transform(d[numericas])).astype(np.float32)  # Imputa y escala las numéricas
        cat = codificador.transform(d[cfg.CATEGORICAS_MODELO])  # Codifica las categóricas
        return np.hstack([num, cat])  # Une ambas partes en una matriz densa

    X_train = transformar(train_t)  # Matriz de entrenamiento
    X_test = transformar(test_t)  # Matriz de prueba
nombres = numericas + codificador.get_feature_names_out(cfg.CATEGORICAS_MODELO).tolist()  # Nombres de las columnas finales
print("X_train:", X_train.shape, " X_test:", X_test.shape)  # Dimensiones
print("Categorías conservadas:", json.dumps(frecuentes, ensure_ascii=False)[:1500])  # Categorías tras agrupar raras
tiempos  # Tiempos de cada etapa

# %%
pd.DataFrame(X_train[:5], columns=nombres)  # Vista de las primeras filas ya transformadas

# %%
pd.DataFrame({"media_train": X_train[:, :len(numericas)].mean(axis=0), "desv_train": X_train[:, :len(numericas)].std(axis=0),  # Verifica que train quede estandarizado
              "media_test": X_test[:, :len(numericas)].mean(axis=0)}, index=numericas)  # En test la media es ≈ 0 pero no exacta (no hay fuga)

# %% [markdown]
# ## Guardado de los datos preprocesados

# %%
np.save(cfg.DIR_SKLEARN_PREP / "X_train.npy", X_train)  # Matriz de entrenamiento
np.save(cfg.DIR_SKLEARN_PREP / "X_test.npy", X_test)  # Matriz de prueba
np.save(cfg.DIR_SKLEARN_PREP / "y_train.npy", train[cfg.COL_TARGET].to_numpy())  # Etiquetas de entrenamiento
np.save(cfg.DIR_SKLEARN_PREP / "y_test.npy", test[cfg.COL_TARGET].to_numpy())  # Etiquetas de prueba
np.save(cfg.DIR_SKLEARN_PREP / "id_train.npy", train[cfg.COL_ID].to_numpy())  # Identificadores de entrenamiento
np.save(cfg.DIR_SKLEARN_PREP / "id_test.npy", test[cfg.COL_ID].to_numpy())  # Identificadores de prueba
joblib.dump({"frecuentes": frecuentes, "imputador": imputador, "escalador": escalador,  # Transformadores ajustados (para LIME)
             "codificador": codificador, "numericas": numericas, "nombres": nombres},  # Metadatos de columnas
            cfg.DIR_SKLEARN_PREP / "transformadores.joblib")  # Archivo de salida
tiempos["total"] = sum(tiempos.values())  # Tiempo total de preprocesamiento
actualizar_json(cfg.RUTA_TIEMPOS_PREP, "sklearn", tiempos)  # Guarda los tiempos para la comparación final
print({k: round(v, 1) for k, v in tiempos.items()})  # Muestra los tiempos en segundos

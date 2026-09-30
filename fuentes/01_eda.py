# %% [markdown]
# # 1. Análisis exploratorio de datos (EDA)
#
# **Objetivo del proyecto:** predecir si un préstamo de Lending Club termina en *default*
# (`Charged Off` = 1) o se paga completamente (`Fully Paid` = 0), comparando seis modelos
# implementados en scikit-learn y en PySpark.
#
# Este notebook cubre las secciones 2.1 a 2.5 del enunciado: carga completa del dataset
# (sin muestreo), análisis unidimensional, bidimensional, multicolinealidad, valores
# faltantes y un resumen ejecutivo con las decisiones que guían el preprocesamiento.
#
# > **Nota sobre la definición del target.** La fórmula del enunciado
# > `df["default"] = df["loan_status"].apply(lambda x: 1 if x == "Charged Off" else 0)`
# > aplicada al archivo completo etiquetaría como "pagados" (0) a los préstamos `Current`,
# > `Late`, `In Grace Period`, etc., que **todavía no tienen un desenlace observado**. Eso sería
# > un error de etiquetado. Por ello primero se conservan solo los préstamos con estado final
# > (`Fully Paid` / `Charged Off`) y después se aplica la fórmula literal. Esto no es un muestreo:
# > se usan **todas** las filas con desenlace conocido (1.345.310, coherente con los "más de
# > 1,3 millones de registros" del enunciado).

# %%
import sys  # Permite modificar la ruta de búsqueda de módulos
sys.path.insert(0, "..")  # Agrega la raíz del proyecto para poder importar src/
import time  # Medición de tiempos
import warnings  # Control de advertencias
import numpy as np  # Cálculo numérico
import pandas as pd  # Manipulación de datos tabulares
import matplotlib.pyplot as plt  # Gráficos
import seaborn as sns  # Gráficos estadísticos
import missingno as msno  # Visualización de valores faltantes
from scipy import stats  # Pruebas estadísticas
from src import config as cfg  # Configuración central del proyecto
from src import preparacion as prep  # Construcción de variables con pandas
from src.utilidades import guardar_figura, info_hardware  # Utilidades auxiliares

warnings.filterwarnings("ignore", category=FutureWarning)  # Oculta avisos de versiones futuras de librerías
pd.set_option("display.max_columns", 60)  # Muestra hasta 60 columnas en las tablas
pd.set_option("display.width", 200)  # Ancho máximo de impresión de tablas
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")  # Formato de los números flotantes
sns.set_theme(style="whitegrid", palette="deep")  # Estilo general de los gráficos
print(info_hardware())  # Muestra el hardware utilizado

# %% [markdown]
# ## 2.2 Carga inicial y visión general
#
# Se carga el CSV comprimido **completo** con pandas y, a modo de comparación, con Spark.

# %%
inicio = time.time()  # Marca de tiempo antes de la carga
df_raw = pd.read_csv(cfg.RUTA_CSV, low_memory=False)  # Lee el archivo completo (pandas descomprime el .gz)
t_pandas = time.time() - inicio  # Tiempo total de carga con pandas
print(f"Tiempo de carga con pandas: {t_pandas:.1f} s")  # Reporta el tiempo
print(f"Dimensión: {df_raw.shape[0]:,} filas × {df_raw.shape[1]} columnas")  # Reporta filas y columnas
print(f"Memoria en uso: {df_raw.memory_usage(deep=True).sum() / 1024**3:.2f} GB")  # Memoria ocupada por el DataFrame

# %%
from src import preparacion_spark as prep_sp  # Funciones de Spark del proyecto
spark = prep_sp.crear_sesion()  # Crea la SparkSession con la configuración del enunciado
inicio = time.time()  # Marca de tiempo antes de la carga
n_spark = prep_sp.leer_csv(spark).count()  # Lee el CSV completo con Spark y fuerza la lectura contando filas
t_spark = time.time() - inicio  # Tiempo total de carga con Spark
print(f"Tiempo de carga con Spark: {t_spark:.1f} s  ({n_spark:,} filas)")  # Reporta tiempo y filas
spark.stop()  # Detiene la sesión para liberar memoria

# %% [markdown]
# **Comentario.** El archivo tiene **2.260.701 filas × 151 columnas** y ocupa ≈ 6,3 GB en
# memoria con pandas, que tardó ≈ 72 s en leerlo. Spark contó las mismas filas en ≈ 17 s: un
# `.gz` no es divisible (se descomprime en un solo hilo), pero Spark no construye objetos de
# Python ni infiere tipos, y para un `count()` apenas necesita tokenizar cada registro. La
# comparación favorece a Spark sobre todo porque su lectura es perezosa: pandas materializa
# todas las columnas tipadas. `head()`/`tail()` confirman una lectura correcta; las dos últimas
# filas son **totales del archivo** (sin `loan_status`) y quedan fuera al filtrar estados.

# %%
df_raw.head()  # Primeras 5 filas para verificar la lectura

# %%
df_raw.tail()  # Últimas 5 filas: las dos últimas son filas de totales, no préstamos

# %%
df_raw.info(verbose=True, show_counts=True)  # Tipos de datos y conteo de no nulos de cada columna

# %%
df_raw.describe().T  # Estadísticas descriptivas de todas las variables numéricas

# %%
estados = df_raw[cfg.COL_ESTADO].value_counts(dropna=False).to_frame("n")  # Frecuencia de cada estado del préstamo
estados["%"] = 100 * estados["n"] / estados["n"].sum()  # Porcentaje de cada estado
estados  # Muestra la tabla

# %% [markdown]
# Solo `Fully Paid` y `Charged Off` representan desenlaces definitivos. `Current` y los
# estados de mora son préstamos vivos; las variantes "Does not meet the credit policy" son
# préstamos bajo una política distinta y `Default` (40 casos) aún no ha sido castigado.

# %% [markdown]
# ### Pre-cálculo de faltantes sobre las 151 columnas
#
# Antes de reducir columnas, se calcula la tasa de nulos de **todas** las variables en los
# préstamos con desenlace final y su relación con el target. Los resultados se presentan en
# la sección de valores faltantes.

# %%
finales = df_raw[cfg.COL_ESTADO].isin(cfg.ESTADOS_FINALES).to_numpy()  # Máscara de préstamos con estado final
nulos_fin = df_raw.isna().to_numpy()[finales]  # Matriz booleana de nulos (filas finales × 151 columnas), sin copiar los datos
y_fin = (df_raw.loc[finales, cfg.COL_ESTADO] == "Charged Off").astype(int).to_numpy()  # Target en ese subconjunto
nulos_todas = pd.DataFrame({"n_nulos": nulos_fin.sum(axis=0)}, index=df_raw.columns)  # Número de nulos por columna
nulos_todas["pct_nulos"] = 100 * nulos_todas["n_nulos"] / finales.sum()  # Porcentaje de nulos por columna
nulos_todas = nulos_todas.sort_values("pct_nulos", ascending=False)  # Ordena de mayor a menor

filas_mcar = []  # Lista con la prueba de independencia nulo-vs-default de cada columna
for col in nulos_todas.index[(nulos_todas["pct_nulos"] > 0) & (nulos_todas["pct_nulos"] < 100)]:  # Columnas con nulos parciales
    es_nulo = nulos_fin[:, df_raw.columns.get_loc(col)]  # Indicador de nulo de la columna
    tabla = pd.crosstab(es_nulo, y_fin)  # Tabla 2×2: nulo/no nulo × default
    chi2, p, _, _ = stats.chi2_contingency(tabla)  # Prueba chi-cuadrado de independencia
    filas_mcar.append({  # Guarda el resultado
        "columna": col,  # Nombre de la columna
        "pct_nulos": nulos_todas.loc[col, "pct_nulos"],  # Porcentaje de nulos
        "tasa_default_si_nulo": y_fin[es_nulo].mean(),  # Tasa de default cuando falta el dato
        "tasa_default_si_presente": y_fin[~es_nulo].mean(),  # Tasa de default cuando el dato está
        "chi2": chi2, "p_valor": p,  # Estadístico y valor p
    })
mcar_todas = pd.DataFrame(filas_mcar).set_index("columna")  # Tabla final de la prueba

anio_fin = pd.to_datetime(df_raw.loc[finales, "issue_d"], format="%b-%Y").dt.year.to_numpy()  # Año de emisión de cada préstamo
cols_con_nulos = nulos_todas.index[(nulos_todas["pct_nulos"] > 0)].tolist()  # Columnas con algún nulo
nulos_por_anio = (pd.DataFrame(nulos_fin, columns=df_raw.columns)[cols_con_nulos]  # Nulos de las columnas con faltantes
                  .groupby(anio_fin).mean())  # Proporción de nulos por año de emisión
nulos_por_fila = nulos_fin.sum(axis=1)  # Número de nulos en cada fila

df = prep.construir_variables(df_raw)  # Filtra estados finales y crea variables derivadas + default
del df_raw, nulos_fin  # Libera el DataFrame de 151 columnas y la matriz de nulos
import gc; gc.collect()  # Fuerza la liberación de memoria
print(df.shape)  # Dimensión del dataset de trabajo
df.head()  # Primeras filas del dataset de trabajo

# %% [markdown]
# ## 2.3 Análisis unidimensional
#
# ### Variables numéricas
#
# Para cada variable se reportan media, mediana, desviación estándar, mínimo, máximo,
# percentiles, IQR, outliers según la regla de Tukey (fuera de [Q1 − 1.5·IQR, Q3 + 1.5·IQR]),
# asimetría, curtosis y la prueba de normalidad de D'Agostino-Pearson (se prefiere a
# Shapiro-Wilk porque esta última no es válida con más de 5.000 observaciones y aquí se usa
# la muestra completa).

# %%
filas = []  # Lista de resúmenes por variable
for col in cfg.NUMERICAS_EDA:  # Recorre cada variable numérica
    x = df[col].dropna()  # Valores no nulos
    q1, q2, q3 = x.quantile([0.25, 0.5, 0.75])  # Cuartiles
    iqr = q3 - q1  # Rango intercuartil
    li, ls = q1 - 1.5 * iqr, q3 + 1.5 * iqr  # Límites de Tukey
    n_out = int(((x < li) | (x > ls)).sum())  # Número de outliers
    filas.append({  # Guarda el resumen de la variable
        "media": x.mean(), "mediana": q2, "desv_est": x.std(),  # Tendencia central y dispersión
        "min": x.min(), "p25": q1, "p50": q2, "p75": q3, "max": x.max(),  # Rango y percentiles
        "IQR": iqr, "lim_inf": li, "lim_sup": ls,  # IQR y límites de outliers
        "n_outliers": n_out, "pct_outliers": 100 * n_out / len(x),  # Cantidad y % de outliers
        "asimetria": stats.skew(x), "curtosis": stats.kurtosis(x),  # Forma de la distribución
        "p_normalidad": stats.normaltest(x).pvalue,  # Prueba de D'Agostino-Pearson
        "pct_nulos": 100 * df[col].isna().mean(),  # Porcentaje de nulos
    })
resumen_num = pd.DataFrame(filas, index=cfg.NUMERICAS_EDA)  # Tabla final
resumen_num  # Muestra la tabla

# %%
fig, axes = plt.subplots(5, 4, figsize=(20, 20))  # Cuadrícula de 5×4 gráficos
for ax, col in zip(axes.ravel(), cfg.NUMERICAS_EDA):  # Un histograma por variable
    x = df[col].dropna()  # Valores no nulos
    x = x[x <= x.quantile(0.995)]  # Recorta el 0,5 % superior solo para que el gráfico sea legible
    ax.hist(x, bins=50, color="#4C72B0", edgecolor="white")  # Histograma de frecuencias
    ax.set_title(col)  # Título con el nombre de la variable
fig.suptitle("Histogramas de las variables numéricas (hasta el percentil 99,5)", y=1.0, fontsize=14)  # Título general
fig.tight_layout()  # Ajusta los márgenes
guardar_figura(fig, "01_histogramas")  # Exporta la figura
plt.show()  # Muestra la figura

# %%
fig, axes = plt.subplots(5, 4, figsize=(20, 16))  # Cuadrícula de 5×4 gráficos
for ax, col in zip(axes.ravel(), cfg.NUMERICAS_EDA):  # Un boxplot por variable
    ax.boxplot(df[col].dropna(), vert=False, flierprops={"markersize": 1, "alpha": 0.2})  # Boxplot horizontal con todos los datos
    ax.set_title(col)  # Título con el nombre de la variable
    ax.set_yticks([])  # Oculta el eje Y (no aporta información)
fig.suptitle("Diagramas de caja (datos completos, incluidos outliers)", y=1.0, fontsize=14)  # Título general
fig.tight_layout()  # Ajusta los márgenes
guardar_figura(fig, "01_boxplots")  # Exporta la figura
plt.show()  # Muestra la figura

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 4))  # Cuatro gráficos en una fila
for i, col in enumerate(cfg.VARIABLES_LOG):  # Para cada variable muy asimétrica
    axes[2 * i].hist(df[col].dropna().clip(upper=df[col].quantile(0.995)), bins=60)  # Distribución original (recortada)
    axes[2 * i].set_title(f"{col} (original)")  # Título
    axes[2 * i + 1].hist(np.log1p(df[col].dropna().clip(lower=0)), bins=60, color="#DD8452")  # Distribución con log1p
    axes[2 * i + 1].set_title(f"log1p({col})  asimetría={stats.skew(np.log1p(df[col].dropna().clip(lower=0))):.2f}")  # Título con la nueva asimetría
fig.tight_layout()  # Ajusta los márgenes
plt.show()  # Muestra la figura

# %% [markdown]
# **Comentario (numéricas).**
#
# * **Ninguna variable es normal** (p < 0.001 en D'Agostino-Pearson para todas; con 1,3 M de
#   observaciones cualquier desviación es significativa). Las asimetrías más extremas son
#   `annual_inc` (46, curtosis ≈ 4.800), `dti` (27), `revol_bal` (13,7) y `pub_rec` (11,6).
#   `log1p` reduce la asimetría de `annual_inc` (46 → −1,9) y `revol_bal` (13,7 → −2,6; la cola
#   izquierda restante se debe a los ceros), por lo que se aplicará en el preprocesamiento (ayuda a la regresión logística, LinearSVC y Naive Bayes;
#   a los árboles les es indiferente). Box-Cox no es aplicable directamente porque hay ceros.
# * **Outliers (regla IQR):** `delinq_2yrs` (19 %), `pub_rec` (17 %) y `pub_rec_bankruptcies`
#   (12 %) tienen IQR = 0, así que cualquier valor distinto de 0 se marca como atípico: no son
#   errores sino eventos raros. `term_meses` solo toma 36 o 60. Sí hay valores **implausibles**:
#   `dti` = 999, `revol_util` = 892 %, `annual_inc` = 11 M. Se conservan (no se permite
#   eliminar filas y los árboles son robustos); el escalado estándar los amortigua para los
#   modelos lineales.
# * **Faltantes:** solo `mths_since_last_delinq` supera el umbral del 30 % (50,5 %); el resto
#   tiene < 6 %. Pese a superar el umbral **no se elimina**: su nulo significa "nunca tuvo
#   morosidad" y es informativo (ver sección de faltantes) → indicador + imputación.

# %% [markdown]
# ### Variables categóricas

# %%
tablas_cat = {}  # Diccionario variable → tabla de frecuencias
for col in cfg.CATEGORICAS_EDA:  # Recorre cada variable categórica
    t = df[col].value_counts(dropna=False).to_frame("frecuencia")  # Frecuencia absoluta (incluye nulos)
    t["porcentaje"] = 100 * t["frecuencia"] / len(df)  # Frecuencia relativa
    t["rara_<1%"] = t["porcentaje"] < 100 * cfg.UMBRAL_CATEGORIA_RARA  # Marca categorías con menos del 1 %
    tablas_cat[col] = t  # Guarda la tabla
    print(f"\n=== {col}: {len(t)} categorías, {int(t['rara_<1%'].sum())} raras, nulos = {100 * df[col].isna().mean():.2f} %")  # Resumen
    display(t.head(15))  # Muestra hasta 15 categorías

# %%
fig, axes = plt.subplots(4, 2, figsize=(18, 22))  # Cuadrícula de 4×2 gráficos
for ax, col in zip(axes.ravel(), cfg.CATEGORICAS_EDA):  # Un gráfico de barras por variable
    t = tablas_cat[col]["porcentaje"].sort_values()  # Porcentajes ordenados
    colores = ["#C44E52" if p < 1 else "#4C72B0" for p in t.values]  # Rojo para categorías raras
    ax.barh(t.index.astype(str), t.values, color=colores)  # Barras horizontales
    ax.set_title(f"{col} (rojo = < 1 %)")  # Título
    ax.set_xlabel("% de préstamos")  # Etiqueta del eje X
    ax.tick_params(axis="y", labelsize=7 if len(t) > 20 else 9)  # Letra más pequeña si hay muchas categorías
fig.tight_layout()  # Ajusta los márgenes
guardar_figura(fig, "01_barras_categoricas")  # Exporta la figura
plt.show()  # Muestra la figura

# %% [markdown]
# **Comentario (categóricas).** No hay nulos en ninguna categórica de trabajo. Categorías con
# menos del 1 %: `grade` G (0,68 %), 10 sub-calificaciones, `home_ownership` ANY/OTHER/NONE
# (478 préstamos en total, **significado redundante**: las tres indican "otra tenencia"),
# seis propósitos (`moving`, `vacation`, `house`, `wedding`, `renewable_energy`, `educational`)
# y 23 estados. Decisión: agrupar en `OTROS` toda categoría con < 1 % **en train** (se aprende
# solo con train para no filtrar información de test). `sub_grade` es una partición fina de
# `grade` (redundante) y `application_type` está muy concentrada (98 % individual).

# %% [markdown]
# ### Variable objetivo (`default`)

# %%
clases = df[cfg.COL_TARGET].value_counts().sort_index().to_frame("n")  # Conteo por clase
clases["%"] = 100 * clases["n"] / clases["n"].sum()  # Porcentaje por clase
clases.index = ["0 = Fully Paid", "1 = Charged Off"]  # Nombres descriptivos
display(clases)  # Muestra la tabla
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Dos gráficos lado a lado
axes[0].bar(clases.index, clases["n"], color=["#4C72B0", "#C44E52"])  # Gráfico de barras
axes[0].set_title("Distribución de la variable objetivo")  # Título
axes[1].pie(clases["n"], labels=clases.index, autopct="%1.1f%%", colors=["#4C72B0", "#C44E52"])  # Gráfico de pastel
guardar_figura(fig, "01_target")  # Exporta la figura
plt.show()  # Muestra la figura
print(f"Razón de desbalance: {clases['n'].iloc[0] / clases['n'].iloc[1]:.2f} : 1")  # Relación entre clases

# %% [markdown]
# **Comentario.** El 80,0 % de los préstamos se pagó y el 20,0 % terminó en default
# (desbalance moderado 4:1). Implicaciones: (1) la partición debe ser **estratificada**;
# (2) la *accuracy* es engañosa (un modelo trivial obtiene 0,80), por lo que el criterio de
# selección es el **AUC ROC** y se reportan también precisión, recall, F1 y AUC-PR; (3) el
# umbral 0,5 tenderá a predecir casi siempre "pagado", así que para las decisiones (McNemar,
# F1) se optimiza el umbral **con datos de train**. No se aplica sobremuestreo: con 269 mil
# defaults la clase minoritaria está bien representada.

# %% [markdown]
# ## 2.4 Análisis bidimensional
#
# ### Numéricas vs. `default`

# %%
fig, axes = plt.subplots(5, 4, figsize=(20, 22))  # Cuadrícula de 5×4 gráficos
for ax, col in zip(axes.ravel(), cfg.NUMERICAS_EDA):  # Un boxplot comparativo por variable
    sns.boxplot(data=df, x=cfg.COL_TARGET, y=col, ax=ax, showfliers=False,  # Boxplot por clase (sin dibujar outliers para legibilidad)
                palette=["#4C72B0", "#C44E52"])  # Colores por clase
    ax.set_title(col)  # Título
fig.suptitle("Boxplots por clase (0 = pagado, 1 = default); outliers omitidos del dibujo", y=1.0, fontsize=14)  # Título general
fig.tight_layout()  # Ajusta los márgenes
guardar_figura(fig, "01_boxplots_por_clase")  # Exporta la figura
plt.show()  # Muestra la figura

# %%
claves = ["int_rate", "fico_range_high", "dti", "annual_inc", "loan_amnt", "term_meses"]  # Variables clave para violines
fig, axes = plt.subplots(2, 3, figsize=(18, 9))  # Cuadrícula de 2×3
for ax, col in zip(axes.ravel(), claves):  # Un violin plot por variable
    datos = df[[col, cfg.COL_TARGET]].dropna()  # Datos sin nulos
    datos = datos[datos[col] <= datos[col].quantile(0.99)]  # Recorta el 1 % superior solo para visualizar
    sns.violinplot(data=datos, x=cfg.COL_TARGET, y=col, ax=ax,  # Violin plot por clase
                   palette=["#4C72B0", "#C44E52"], cut=0)  # Colores y sin extrapolar la densidad
    ax.set_title(col)  # Título
fig.tight_layout()  # Ajusta los márgenes
guardar_figura(fig, "01_violines")  # Exporta la figura
plt.show()  # Muestra la figura

# %%
filas = []  # Lista de resultados por variable
for col in cfg.NUMERICAS_EDA:  # Recorre cada variable numérica
    d = df[[col, cfg.COL_TARGET]].dropna()  # Pares sin nulos
    x0 = d.loc[d[cfg.COL_TARGET] == 0, col]  # Valores de la clase 0
    x1 = d.loc[d[cfg.COL_TARGET] == 1, col]  # Valores de la clase 1
    t_stat, p_t = stats.ttest_ind(x1, x0, equal_var=False)  # Prueba t de Welch (varianzas distintas)
    u_stat, p_u = stats.mannwhitneyu(x1, x0, alternative="two-sided")  # Prueba de Mann-Whitney (no paramétrica)
    r_pb, p_pb = stats.pointbiserialr(d[cfg.COL_TARGET], d[col])  # Correlación punto-biserial
    filas.append({  # Guarda el resultado
        "media_0": x0.mean(), "media_1": x1.mean(), "dif_medias": x1.mean() - x0.mean(),  # Medias y diferencia
        "mediana_0": x0.median(), "mediana_1": x1.median(),  # Medianas por clase
        "p_t_welch": p_t, "p_mann_whitney": p_u,  # Valores p de ambas pruebas
        "r_punto_biserial": r_pb, "abs_r": abs(r_pb),  # Correlación y su valor absoluto
    })
biv_num = pd.DataFrame(filas, index=cfg.NUMERICAS_EDA).sort_values("abs_r", ascending=False)  # Ordena por fuerza de asociación
biv_num  # Muestra la tabla

# %%
fig, ax = plt.subplots(figsize=(10, 7))  # Figura para la correlación punto-biserial
biv_num["r_punto_biserial"].sort_values().plot.barh(ax=ax, color=["#C44E52" if v > 0 else "#4C72B0" for v in biv_num["r_punto_biserial"].sort_values()])  # Barras por signo
ax.set_title("Correlación punto-biserial con default")  # Título
ax.axvline(0, color="black", lw=0.8)  # Línea de referencia en cero
guardar_figura(fig, "01_punto_biserial")  # Exporta la figura
plt.show()  # Muestra la figura

# %% [markdown]
# **Comentario.** Todas las diferencias de medias son significativas (p < 0.001 en Welch y
# Mann-Whitney), algo esperable con 1,3 M de filas; por eso se ordena por la **magnitud** de la
# correlación punto-biserial. Las numéricas más asociadas al default son `int_rate`
# (r = 0,26; media 15,7 % en defaults vs 12,6 %), `term_meses` (0,18: los préstamos a 60
# meses son más riesgosos), `fico_range_high` (−0,13; 10 puntos menos en defaults), `dti`
# (0,08), `mort_acc` (−0,08), `loan_amnt`, `inq_last_6mths` y `revol_util` (≈ 0,06). El
# resto tiene |r| < 0,05: aportan poco de forma individual, aunque pueden interactuar en
# modelos no lineales.

# %% [markdown]
# ### Categóricas vs. `default`

# %%
def v_cramer(x, y):  # Coeficiente V de Cramér entre dos variables categóricas
    tabla = pd.crosstab(x, y)  # Tabla de contingencia
    chi2 = stats.chi2_contingency(tabla, correction=False)[0]  # Estadístico chi-cuadrado
    n = tabla.values.sum()  # Total de observaciones
    k = min(tabla.shape) - 1  # Grados de libertad mínimos
    return np.sqrt(chi2 / (n * k)) if k > 0 else 0.0  # V = sqrt(χ² / (n·k))

filas = []  # Resultados de la prueba chi-cuadrado
tasas_cat = {}  # Tasa de default por categoría
for col in cfg.CATEGORICAS_EDA:  # Recorre cada categórica
    x = df[col].fillna(cfg.CATEGORIA_FALTANTE)  # Nulos como categoría propia
    tabla = pd.crosstab(x, df[cfg.COL_TARGET])  # Tabla de contingencia
    chi2, p, gl, _ = stats.chi2_contingency(tabla)  # Prueba de independencia
    tasa = pd.crosstab(x, df[cfg.COL_TARGET], normalize="index")[1].sort_values(ascending=False)  # Proporción de default por categoría
    tasas_cat[col] = tasa  # Guarda las tasas
    filas.append({"chi2": chi2, "gl": gl, "p_valor": p, "V_cramer": v_cramer(x, df[cfg.COL_TARGET]),  # Estadísticos
                  "categoria_mayor_tasa": tasa.index[0], "mayor_tasa": tasa.iloc[0],  # Categoría más riesgosa
                  "categoria_menor_tasa": tasa.index[-1], "menor_tasa": tasa.iloc[-1]})  # Categoría menos riesgosa
biv_cat = pd.DataFrame(filas, index=cfg.CATEGORICAS_EDA).sort_values("V_cramer", ascending=False)  # Tabla ordenada
biv_cat  # Muestra la tabla

# %%
for col in ["grade", "home_ownership", "verification_status", "purpose", "initial_list_status", "application_type"]:  # Tablas cruzadas de las categóricas más legibles
    ct = pd.crosstab(df[col].fillna(cfg.CATEGORIA_FALTANTE), df[cfg.COL_TARGET], margins=True)  # Frecuencias con totales
    ct["tasa_default_%"] = 100 * ct[1] / ct["All"]  # Tasa de impago por categoría
    display(ct.sort_values("tasa_default_%", ascending=False))  # Muestra la tabla ordenada por riesgo

# %%
fig, axes = plt.subplots(4, 2, figsize=(18, 22))  # Cuadrícula de 4×2
tasa_global = df[cfg.COL_TARGET].mean()  # Tasa de default global
for ax, col in zip(axes.ravel(), cfg.CATEGORICAS_EDA):  # Un gráfico de barras apiladas por variable
    prop = pd.crosstab(df[col].fillna(cfg.CATEGORIA_FALTANTE), df[cfg.COL_TARGET], normalize="index")  # Proporciones por categoría
    prop = prop.sort_values(1)  # Ordena por tasa de default
    prop.plot.barh(stacked=True, ax=ax, color=["#4C72B0", "#C44E52"], width=0.85, legend=False)  # Barras apiladas 0/1
    ax.axvline(1 - tasa_global, color="black", ls="--", lw=1)  # Línea de la tasa global
    ax.set_title(f"{col}: proporción de default por categoría")  # Título
    ax.tick_params(axis="y", labelsize=7 if len(prop) > 20 else 9)  # Letra más pequeña si hay muchas categorías
fig.tight_layout()  # Ajusta los márgenes
guardar_figura(fig, "01_barras_apiladas")  # Exporta la figura
plt.show()  # Muestra la figura

# %% [markdown]
# **Comentario.** La prueba chi-cuadrado rechaza la independencia para todas las categóricas
# (p < 0.001), pero la fuerza de asociación (V de Cramér) es muy distinta: `sub_grade`
# (0,27) y `grade` (0,26) dominan — la tasa de default crece de forma monótona de 6 % en A a
# 50 % en G —; siguen `verification_status` (0,09; los ingresos *verificados* tienen más
# default, 23,9 %, porque Lending Club verifica sobre todo a los solicitantes más riesgosos),
# `home_ownership` (RENT 23,2 %), `purpose` (small_business 29,7 %) y `addr_state` (0,05).
# `initial_list_status` y `application_type` son casi independientes del target (V < 0,02).

# %% [markdown]
# ### Multicolinealidad

# %%
corr = df[cfg.NUMERICAS_EDA].corr(method="pearson")  # Matriz de correlación de Pearson
fig, ax = plt.subplots(figsize=(15, 12))  # Figura grande para el mapa de calor
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7})  # Mapa de calor
ax.set_title("Correlación de Pearson entre variables numéricas")  # Título
guardar_figura(fig, "01_correlacion")  # Exporta la figura
plt.show()  # Muestra la figura
pares = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()  # Pares únicos (triángulo superior)
pares.reindex(pares.abs().sort_values(ascending=False).index).head(12).to_frame("r")  # Pares más correlacionados

# %%
matriz_v = pd.DataFrame(index=cfg.CATEGORICAS_EDA, columns=cfg.CATEGORICAS_EDA, dtype=float)  # Matriz vacía de V de Cramér
for a in cfg.CATEGORICAS_EDA:  # Primera variable del par
    for b in cfg.CATEGORICAS_EDA:  # Segunda variable del par
        matriz_v.loc[a, b] = 1.0 if a == b else v_cramer(df[a].fillna(cfg.CATEGORIA_FALTANTE), df[b].fillna(cfg.CATEGORIA_FALTANTE))  # V de Cramér del par
fig, ax = plt.subplots(figsize=(9, 7))  # Figura para el mapa de calor
sns.heatmap(matriz_v, annot=True, fmt=".2f", cmap="Purples", vmin=0, vmax=1, ax=ax)  # Mapa de calor de V de Cramér
ax.set_title("V de Cramér entre variables categóricas")  # Título
guardar_figura(fig, "01_cramer")  # Exporta la figura
plt.show()  # Muestra la figura

# %%
fig, ax = plt.subplots(figsize=(10, 5))  # Relación entre sub-calificación y tasa de interés
orden = sorted(df["sub_grade"].unique())  # Sub-calificaciones en orden A1…G5
sns.boxplot(data=df, x="sub_grade", y="int_rate", order=orden, showfliers=False, ax=ax, color="#8172B3")  # Tasa de interés por sub-calificación
ax.tick_params(axis="x", rotation=90)  # Rota las etiquetas
ax.set_title("int_rate por sub_grade: la tasa de interés es casi una función de la calificación")  # Título
plt.show()  # Muestra la figura
print("Correlación de Spearman entre fico_range_high e int_rate:", round(stats.spearmanr(df["fico_range_high"], df["int_rate"])[0], 3))  # Redundancia sugerida por el enunciado

# %% [markdown]
# **Comentario (multicolinealidad).** Pares con |r| > 0,7: `fico_range_low`–`fico_range_high`
# (r = 1,00: el rango siempre mide 4 puntos), `loan_amnt`–`installment` (0,95: la cuota es
# función del monto, la tasa y el plazo) y `open_acc`–`total_acc` (0,70, justo en el límite).
# Entre categóricas, `grade`–`sub_grade` tiene V ≈ 1 (una determina a la otra) y la tasa de
# interés es prácticamente una función de la sub-calificación. La redundancia sugerida por el
# enunciado `fico_range_high`–`int_rate` es moderada (ρ = −0,42), no suficiente para excluir.
#
# **Decisión:** se excluyen `fico_range_low`, `installment` y `sub_grade`; se conservan
# `open_acc` y `total_acc` (r en el límite y significado distinto) e `int_rate` junto a
# `grade` (la tasa aporta variación dentro de cada grado).

# %% [markdown]
# ### Visualizaciones avanzadas
#
# El *pairplot* y la matriz de dispersión dibujan cada punto, por lo que (como permite el
# enunciado) se usa una **muestra aleatoria de 5.000 filas solo para estos dos gráficos**.
# Ningún cálculo ni modelo usa la muestra.

# %%
top_vars = ["int_rate", "fico_range_high", "dti", "annual_inc", "term_meses"]  # Variables más relevantes
muestra_viz = df[top_vars + [cfg.COL_TARGET]].dropna().sample(5000, random_state=cfg.SEMILLA)  # Muestra solo para visualización
muestra_viz["annual_inc"] = np.log1p(muestra_viz["annual_inc"])  # Escala logarítmica para el ingreso
g = sns.pairplot(muestra_viz, hue=cfg.COL_TARGET, palette=["#4C72B0", "#C44E52"], plot_kws={"s": 6, "alpha": 0.4}, corner=True)  # Pairplot por clase
g.fig.suptitle("Pairplot (muestra de 5.000, annual_inc en log1p)", y=1.02)  # Título
plt.show()  # Muestra la figura

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 9))  # KDE comparativos por clase (datos completos)
for ax, col in zip(axes.ravel(), ["int_rate", "fico_range_high", "dti", "annual_inc", "revol_util", "antiguedad_credito_meses"]):  # Variables relevantes
    d = df[[col, cfg.COL_TARGET]].dropna()  # Datos sin nulos
    if col == "annual_inc":  # El ingreso se grafica en escala log
        d[col] = np.log1p(d[col])  # Transformación log1p
    d = d[d[col].between(d[col].quantile(0.005), d[col].quantile(0.995))]  # Recorta colas extremas solo para visualizar
    for clase, color in [(0, "#4C72B0"), (1, "#C44E52")]:  # Una curva por clase
        sns.kdeplot(d.loc[d[cfg.COL_TARGET] == clase, col], ax=ax, color=color, fill=True, alpha=0.3, label=f"default={clase}", bw_adjust=1.5)  # Densidad
    ax.set_title(col if col != "annual_inc" else "log1p(annual_inc)")  # Título
    ax.legend()  # Leyenda
fig.tight_layout()  # Ajusta los márgenes
guardar_figura(fig, "01_kde")  # Exporta la figura
plt.show()  # Muestra la figura

# %%
colores = muestra_viz[cfg.COL_TARGET].map({0: "#4C72B0", 1: "#C44E52"})  # Color de cada punto según la clase
pd.plotting.scatter_matrix(muestra_viz[top_vars], c=colores, figsize=(13, 13), s=6, alpha=0.4, diagonal="hist")  # Matriz de dispersión de 5 variables
plt.suptitle("Matriz de dispersión (muestra de 5.000)", y=0.92)  # Título
plt.show()  # Muestra la figura

# %% [markdown]
# **Comentario.** Las densidades muestran solapamiento amplio entre clases: ninguna variable
# separa por sí sola. Las diferencias más visibles son el desplazamiento de `int_rate` hacia
# la derecha y de `fico_range_high` hacia la izquierda en los defaults. En el *pairplot* y la
# matriz de dispersión no aparecen fronteras no lineales claras; sí se aprecia la estructura
# discreta de `term_meses` y de FICO, y la relación decreciente entre FICO y tasa de interés.

# %% [markdown]
# ## Valores faltantes (análisis detallado)

# %%
print(f"Columnas sin nulos: {(nulos_todas['pct_nulos'] == 0).sum()} | con nulos: {(nulos_todas['pct_nulos'] > 0).sum()} | "
      f"> 30 %: {(nulos_todas['pct_nulos'] > 30).sum()} | > 70 %: {(nulos_todas['pct_nulos'] > 70).sum()}")  # Resumen de las 151 columnas
nulos_todas.head(40)  # Las 40 columnas con más nulos

# %%
fig, ax = plt.subplots(figsize=(10, 22))  # Figura alta para las columnas con nulos
d = nulos_todas.loc[nulos_todas["pct_nulos"] > 0, "pct_nulos"].sort_values()  # Columnas con nulos
ax.barh(d.index, d.values, color=["#C44E52" if v > 70 else "#DD8452" if v > 30 else "#4C72B0" for v in d.values])  # Barras por severidad
ax.axvline(30, color="gray", ls="--"); ax.axvline(70, color="black", ls="--")  # Umbrales 30 % y 70 %
ax.set_xlabel("% de nulos"); ax.tick_params(axis="y", labelsize=7)  # Etiquetas
ax.set_title("Porcentaje de nulos por columna (préstamos con desenlace final)")  # Título
guardar_figura(fig, "01_nulos_columnas")  # Exporta la figura
plt.show()  # Muestra la figura

# %%
fig, ax = plt.subplots(figsize=(14, 18))  # Mapa de calor del patrón de nulos a lo largo del tiempo
sns.heatmap(nulos_por_anio.T, cmap="Reds", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "proporción de nulos"})  # Filas = columnas, columnas = año
ax.set_xlabel("Año de emisión"); ax.tick_params(axis="y", labelsize=6)  # Etiquetas
ax.set_title("Patrón de valores faltantes por año de emisión (todas las filas)")  # Título
guardar_figura(fig, "01_nulos_por_anio")  # Exporta la figura
plt.show()  # Muestra la figura

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # Concentración de nulos por fila + missingno
axes[0].hist(nulos_por_fila, bins=range(0, int(nulos_por_fila.max()) + 2), color="#4C72B0")  # Número de nulos por fila
axes[0].set_title("Número de columnas nulas por fila (151 columnas)"); axes[0].set_xlabel("nulos en la fila")  # Títulos
msno.bar(df[cfg.NUMERICAS_EDA + cfg.CATEGORICAS_EDA], ax=axes[1], fontsize=8, color="#4C72B0")  # Completitud de las variables de trabajo
axes[1].set_title("Completitud de las variables de trabajo (missingno)")  # Título
plt.show()  # Muestra la figura

# %%
con_nulos = [c for c in cfg.NUMERICAS_EDA + cfg.CATEGORICAS_EDA if df[c].isna().any()]  # Variables de trabajo con algún nulo
msno.heatmap(df[con_nulos], figsize=(9, 6), fontsize=9)  # Correlación de nulidad entre variables (datos completos)
plt.title("Correlación de nulidad (missingno)")  # Título
plt.show()  # Muestra la figura

# %%
mcar_todas["relacion_con_default"] = np.where(mcar_todas["p_valor"] < cfg.ALFA, "NO MCAR (p < 0.05)", "compatible con MCAR")  # Interpretación
mcar_trabajo = mcar_todas.reindex([c for c in ["emp_length", "mths_since_last_delinq", "mort_acc", "revol_util", "dti", "pub_rec_bankruptcies", "earliest_cr_line", "inq_last_6mths"] if c in mcar_todas.index])  # Columnas de trabajo con nulos
display(mcar_trabajo)  # Prueba sobre las variables de trabajo
print("Resumen sobre las", len(mcar_todas), "columnas con nulos parciales:")  # Resumen general
mcar_todas["relacion_con_default"].value_counts()  # Cuántas columnas muestran nulos relacionados con el target

# %% [markdown]
# La prueba anterior contrasta, para cada columna, si **estar nulo** es independiente del
# target (chi-cuadrado sobre la tabla 2×2 nulo × default). Un valor p < 0.05 indica que
# los nulos **no son completamente al azar (no MCAR)**: su presencia aporta información sobre
# el riesgo. Con 1,3 millones de filas casi cualquier diferencia resulta significativa, por lo
# que también se reporta la magnitud (tasa de default con y sin dato).

# %%
plan = pd.DataFrame({  # Plan de tratamiento de nulos para las variables de trabajo
    "pct_nulos": [100 * df[c].isna().mean() for c in cfg.NUMERICAS_EDA + cfg.CATEGORICAS_EDA],  # % de nulos
}, index=cfg.NUMERICAS_EDA + cfg.CATEGORICAS_EDA)  # Índice = variables
plan = plan[plan["pct_nulos"] > 0].sort_values("pct_nulos", ascending=False)  # Solo las que tienen nulos
plan["tratamiento"] = [  # Regla de tratamiento por variable
    "Indicador de nulo + imputación con mediana de train (nulo = sin morosidad registrada)" if c == "mths_since_last_delinq"  # Caso no MCAR con 30–70 % de nulos
    else "Indicador de nulo + imputación con mediana de train (nulos no MCAR)" if c in cfg.VARIABLES_CON_INDICADOR  # Pocos nulos pero informativos
    else "Imputación con mediana de train" if c in cfg.NUMERICAS_EDA  # Numéricas con pocos nulos
    else "Categoría 'DESCONOCIDO'"  # Categóricas
    for c in plan.index]  # Recorre cada variable
plan  # Muestra el plan

# %% [markdown]
# Para las 151 columnas originales el criterio es: **> 70 % de nulos → eliminar**; 30–70 % →
# imputación avanzada o indicador; < 30 % → mediana/moda o categoría "DESCONOCIDO". Además,
# se excluyen **por fuga de información** todas las variables registradas después de la
# originación (pagos, recuperaciones, `last_*`, `hardship_*`, `settlement_*`, `out_prncp`, etc.),
# porque solo se conocen cuando el préstamo ya está en curso o terminado.

# %% [markdown]
# **Comentario (faltantes).** De las 151 columnas, 46 no tienen nulos, 58 superan el 30 % y
# 42 el 70 % (datos de co-solicitante, *hardship*, *settlement*, `desc`, `member_id` al 100 %).
# El mapa por año muestra que los nulos son **sistemáticos por época**: muchas variables de
# buró (`open_acc_6m`, `il_util`, `mths_since_rcnt_il`…) solo existen desde finales de 2015
# (95 % nulos en 2015, ≈ 0 % desde 2016), las de co-solicitante aparecen desde 2016 (solicitudes
# conjuntas) y `mort_acc` falta en todos los préstamos anteriores a 2012. Es decir,
# los nulos se concentran en columnas y periodos, no en filas aisladas. En 95 de las 103
# columnas con nulos parciales la probabilidad de faltar está relacionada con el default
# (**no MCAR**). Entre las variables de trabajo: sin `emp_length` el default sube a 26,9 %
# (vs 19,5 %), sin `mort_acc` baja a 14,6 % (préstamos antiguos) y sin
# `mths_since_last_delinq` es 19,3 % (vs 20,7 %). Por eso esas tres reciben un **indicador de
# nulo** además de la imputación con la mediana de train.

# %% [markdown]
# ## 2.5 Resumen ejecutivo del EDA
#
# | Aspecto | Hallazgo | Decisión para el preprocesamiento |
# |---|---|---|
# | Tamaño y target | 2.260.701 filas; 1.345.310 con desenlace final; 20,0 % default (4:1) | Filtrar a `Fully Paid`/`Charged Off`; partición 80/20 estratificada; AUC como criterio; umbral optimizado en train |
# | Calidad | `id` sin duplicados; 2 filas de totales al final; valores implausibles (`dti` = 999, `revol_util` = 892) | Se conservan todas las filas (sin muestreo ni eliminación); escalado estándar |
# | Asimetría | Ninguna variable normal; `annual_inc` y `revol_bal` extremadamente asimétricas | `log1p` en ambas |
# | Predictores más prometedores | `grade`/`int_rate`, `term`, FICO, `dti`, `verification_status`, `home_ownership`, `purpose`, `revol_util`, `inq_last_6mths`, `mort_acc` | Se incluyen todas las variables conocidas al originar el préstamo (17 numéricas + 7 categóricas) |
# | Multicolinealidad | FICO bajo/alto (r = 1), monto/cuota (0,95), `grade`/`sub_grade` (V ≈ 1) | Excluir `fico_range_low`, `installment`, `sub_grade` |
# | Categorías raras | < 1 % en `grade`, `home_ownership`, `purpose`, `addr_state` | Agrupar en `OTROS` (aprendido en train) |
# | Faltantes | 42 columnas > 70 %; nulos por época; `mths_since_last_delinq` 50 %; nulos no MCAR | Eliminar columnas > 70 % y de fuga; indicador + mediana en 3 variables |
# | Fuga de información | Pagos, recuperaciones, `last_*`, *hardship*, *settlement* | Excluidas |
# | Balanceo | Desbalance moderado | Sin remuestreo; métricas y umbral adecuados |


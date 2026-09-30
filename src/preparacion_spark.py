"""Construcción de variables y preprocesamiento con PySpark.

Replica en Spark las mismas reglas de ``src/preparacion.py`` sin transferir
nunca el dataset al driver (no se usa .toPandas() ni .collect() sobre los datos).
"""

from pyspark.ml import Pipeline  # Encadena las etapas de preprocesamiento
from pyspark.ml.feature import (  # Transformadores de pyspark.ml
    Imputer,  # Imputación de nulos numéricos (mediana)
    OneHotEncoder,  # Codificación one-hot
    StandardScaler,  # Escalado a media 0 y desviación 1
    StringIndexer,  # Convierte texto en índices numéricos
    VectorAssembler,  # Une columnas en un vector de características
)
from pyspark.sql import SparkSession  # Punto de entrada de Spark
from pyspark.sql import functions as F  # Funciones de columnas de Spark SQL

from src import config as cfg  # Configuración central del proyecto


def crear_sesion(memoria=cfg.MEMORIA_SPARK, particiones=cfg.PARTICIONES_SPARK, app="LendingClub_Optimized"):  # Crea la SparkSession del proyecto
    """Configuración de la sección 4.3 en modo local; `particiones` = 400 reproduce la configuración literal del enunciado."""
    spark = (  # Construye la sesión con el patrón builder
        SparkSession.builder  # Inicia el constructor de la sesión
        .appName(app)  # Nombre de la aplicación
        .master(f"local[{cfg.NUCLEOS_SPARK}]")  # Modo local con un hilo por núcleo físico (6)
        .config("spark.sql.shuffle.partitions", str(particiones))  # Particiones tras un shuffle
        .config("spark.default.parallelism", str(particiones))  # Paralelismo por defecto de los RDD
        .config("spark.executor.memory", memoria)  # Memoria del ejecutor (en modo local la usa el driver)
        .config("spark.driver.memory", memoria)  # Memoria del driver (donde corre todo en modo local)
        .config("spark.memory.fraction", 0.8)  # Fracción del heap para ejecución + almacenamiento
        .config("spark.memory.storageFraction", 0.3)  # Parte de esa fracción reservada para caché
        .config("spark.driver.maxResultSize", "2g")  # Límite de resultados enviados al driver
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")  # Arrow de Spark 3.5 no es compatible con Java 21 (JDK de este equipo)
        .getOrCreate()  # Crea la sesión (o reutiliza la existente)
    )
    spark.sparkContext.setLogLevel("ERROR")  # Muestra solo errores en el registro de Spark
    return spark  # Devuelve la sesión


def leer_csv(spark, ruta=cfg.RUTA_CSV):  # Lee el CSV completo (sin muestreo) con Spark
    """Lee el .csv.gz completo; multiLine/escape son necesarios por los textos con comillas y saltos de línea."""
    return spark.read.csv(  # Lector CSV de Spark
        str(ruta),  # Ruta del archivo comprimido
        header=True,  # La primera fila contiene los nombres de columna
        inferSchema=False,  # Todo se lee como texto y se convierte explícitamente (más rápido y controlado)
        multiLine=True,  # Algunos campos de texto contienen saltos de línea
        escape='"',  # Las comillas dentro de campos se escapan con comillas dobles
    )


def construir_variables(df):  # Versión Spark de preparacion.construir_variables
    """Filtra estados finales, crea default y las variables derivadas."""
    df = df.filter(F.col(cfg.COL_ESTADO).isin(cfg.ESTADOS_FINALES))  # Solo préstamos Fully Paid / Charged Off
    df = df.withColumn(cfg.COL_TARGET, F.when(F.col(cfg.COL_ESTADO) == "Charged Off", 1).otherwise(0))  # Target binario
    df = df.withColumn(cfg.COL_ID, F.col(cfg.COL_ID).cast("long"))  # El id pasa a entero largo
    df = df.withColumn("term_meses", F.regexp_extract("term", r"(\d+)", 1).cast("double"))  # " 36 months" → 36
    emp = F.col("emp_length")  # Referencia a la columna de antigüedad laboral
    df = df.withColumn(  # Antigüedad laboral numérica
        "emp_length_anios",  # Nombre de la nueva columna
        F.when(emp.isNull() | (emp == "n/a"), F.lit(None).cast("double"))  # Nulos se mantienen nulos
        .when(emp.contains("<"), F.lit(0.0))  # "< 1 year" → 0
        .otherwise(F.regexp_extract(emp, r"(\d+)", 1).cast("double")),  # Resto: se extrae el número
    )
    emision = F.to_date(F.concat(F.lit("01-"), F.col("issue_d")), "dd-MMM-yyyy")  # Fecha de emisión (día 1 del mes)
    primera = F.to_date(F.concat(F.lit("01-"), F.col("earliest_cr_line")), "dd-MMM-yyyy")  # Fecha de la primera línea
    df = df.withColumn(  # Antigüedad crediticia en meses
        "antiguedad_credito_meses",  # Nombre de la nueva columna
        ((F.year(emision) - F.year(primera)) * 12 + (F.month(emision) - F.month(primera))).cast("double"),  # Años*12 + meses
    )
    numericas_originales = [c for c in cfg.NUMERICAS_EDA if c not in ("term_meses", "emp_length_anios", "antiguedad_credito_meses")]  # Numéricas que vienen como texto
    for c in numericas_originales:  # Recorre esas columnas
        df = df.withColumn(c, F.col(c).cast("double"))  # Convierte texto a número (vacío → nulo)
    columnas = [cfg.COL_ID] + cfg.NUMERICAS_EDA + cfg.CATEGORICAS_EDA + [cfg.COL_TARGET]  # Columnas finales
    return df.select(*columnas)  # Devuelve solo las columnas necesarias


def transformar_numericas(df):  # Versión Spark de preparacion.transformar_numericas
    """log1p en variables asimétricas + indicadores de nulo."""
    for c in cfg.VARIABLES_LOG:  # Recorre las variables asimétricas
        df = df.withColumn(c, F.log1p(F.greatest(F.col(c), F.lit(0.0))))  # log(1 + max(x, 0))
    for col, ind in zip(cfg.VARIABLES_CON_INDICADOR, cfg.INDICADORES_NULO):  # Recorre las variables con nulos no MCAR
        df = df.withColumn(ind, F.col(col).isNull().cast("double"))  # 1 si el dato falta
    return df  # Devuelve el DataFrame transformado


def categorias_frecuentes(train, columnas, umbral=cfg.UMBRAL_CATEGORIA_RARA):  # Aprende categorías frecuentes en train
    """Solo se transfieren al driver los conteos agregados (decenas de filas), nunca los datos."""
    n = train.count()  # Número de filas de entrenamiento
    resultado = {}  # Diccionario columna → categorías frecuentes
    for col in columnas:  # Recorre cada categórica
        conteos = (  # Tabla de frecuencias agregada en Spark
            train.select(F.coalesce(F.col(col), F.lit(cfg.CATEGORIA_FALTANTE)).alias(col))  # Nulos → "DESCONOCIDO"
            .groupBy(col).count()  # Cuenta por categoría
            .filter(F.col("count") / n >= umbral)  # Conserva las que superan el umbral
            .collect()  # Trae al driver solo la tabla agregada (máx. ~50 filas)
        )
        resultado[col] = sorted(r[col] for r in conteos)  # Guarda la lista ordenada de categorías frecuentes
    return resultado  # Devuelve el diccionario


def agrupar_categorias(df, frecuentes):  # Aplica la agrupación de categorías raras
    """Nulos → 'DESCONOCIDO'; categorías no frecuentes → 'OTROS'."""
    for col, cats in frecuentes.items():  # Recorre cada categórica
        valor = F.coalesce(F.col(col), F.lit(cfg.CATEGORIA_FALTANTE))  # Sustituye nulos
        df = df.withColumn(col, F.when(valor.isin(cats), valor).otherwise(F.lit(cfg.CATEGORIA_RARA)))  # Agrupa raras
    return df  # Devuelve el DataFrame transformado


def pipeline_preprocesamiento():  # Define las etapas que aprenden parámetros (se ajustan solo con train)
    """Imputer → VectorAssembler → StandardScaler (numéricas) + StringIndexer → OneHotEncoder (categóricas)."""
    numericas = cfg.NUMERICAS_MODELO + cfg.INDICADORES_NULO  # Numéricas finales más los indicadores de nulo
    imputadas = [f"{c}_imp" for c in numericas]  # Nombres de salida del imputador
    imputer = Imputer(inputCols=numericas, outputCols=imputadas, strategy="median")  # Imputa con la mediana de train
    ensam_num = VectorAssembler(inputCols=imputadas, outputCol="num_vec")  # Vector de numéricas imputadas
    escalador = StandardScaler(inputCol="num_vec", outputCol="num_esc", withMean=True, withStd=True)  # Estandariza como sklearn
    indexados = [f"{c}_idx" for c in cfg.CATEGORICAS_MODELO]  # Nombres de salida del StringIndexer
    ohe_cols = [f"{c}_ohe" for c in cfg.CATEGORICAS_MODELO]  # Nombres de salida del OneHotEncoder
    indexador = StringIndexer(  # Convierte categorías en índices
        inputCols=cfg.CATEGORICAS_MODELO,  # Columnas categóricas
        outputCols=indexados,  # Columnas de salida
        handleInvalid="keep",  # Categorías no vistas en train reciben un índice extra
        stringOrderType="alphabetAsc",  # Orden alfabético, igual que OneHotEncoder de sklearn
    )
    ohe = OneHotEncoder(inputCols=indexados, outputCols=ohe_cols, dropLast=False)  # Una columna por categoría (como sklearn)
    ensam_final = VectorAssembler(inputCols=["num_esc"] + ohe_cols, outputCol="features")  # Vector final de características
    return Pipeline(stages=[imputer, ensam_num, escalador, indexador, ohe, ensam_final])  # Pipeline de preprocesamiento

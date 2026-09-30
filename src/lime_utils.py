"""Utilidades de LIME compartidas por los notebooks 04 (PySpark) y 06 (scikit-learn).

LIME perturba las variables en su espacio original (antes de escalar y codificar) para que
las explicaciones sean legibles. Ambos entornos usan exactamente el mismo explicador, las
mismas instancias y la misma semilla, de modo que reciben las mismas perturbaciones.
"""

import joblib  # Carga de los transformadores ajustados en el notebook 02
import numpy as np  # Cálculo numérico
import pandas as pd  # Manipulación de datos
from lime.lime_tabular import LimeTabularExplainer  # Explicador de LIME para datos tabulares

from src import config as cfg  # Configuración central
from src import preparacion as prep  # Agrupación de categorías con pandas

VARIABLES = cfg.NUMERICAS_MODELO + cfg.CATEGORICAS_MODELO  # Variables del espacio de explicación
N_MUESTRAS = 2000  # Número de perturbaciones por explicación
N_VARIABLES = 10  # Número de variables mostradas en cada explicación
RUTA_EXPLICACIONES_SP = cfg.DIR_SALIDAS / "lime_pyspark.json"  # Explicaciones de PySpark (generadas en el notebook 04)


class EspacioLime:  # Encapsula los datos y las conversiones entre el espacio original y el de LIME
    def __init__(self):  # Carga datos, partición y categorías aprendidas en train
        datos = pd.read_parquet(cfg.RUTA_MODELADO).merge(pd.read_parquet(cfg.RUTA_PARTICION), on=cfg.COL_ID)  # Datos + split
        self.train = datos[datos["split"] == "train"].reset_index(drop=True)  # Conjunto de entrenamiento
        self.test = datos[datos["split"] == "test"].reset_index(drop=True)  # Conjunto de prueba
        self.frecuentes = joblib.load(cfg.DIR_SKLEARN_PREP / "transformadores.joblib")["frecuentes"]  # Categorías frecuentes de train
        self.cats = {c: sorted(set(self.frecuentes[c]) | {cfg.CATEGORIA_RARA}) for c in cfg.CATEGORICAS_MODELO}  # Categorías posibles
        self.medianas = self.train[cfg.NUMERICAS_MODELO].median()  # Medianas de train para nulos no informativos
        self.X_train = self.a_lime(self.train)  # Train en el espacio de LIME (estadísticas del explicador)

    def a_lime(self, d):  # DataFrame original → matriz numérica de LIME
        d = prep.agrupar_categorias(d, self.frecuentes)  # Nulos → DESCONOCIDO, raras → OTROS
        out = pd.DataFrame(index=d.index)  # DataFrame de salida
        for c in cfg.NUMERICAS_MODELO:  # Recorre las numéricas
            out[c] = d[c].fillna(-1.0) if c in cfg.VARIABLES_CON_INDICADOR else d[c].fillna(self.medianas[c])  # −1 = faltante informativo
        for c in cfg.CATEGORICAS_MODELO:  # Recorre las categóricas
            codigos = {v: i for i, v in enumerate(self.cats[c])}  # Texto → código entero
            out[c] = d[c].map(codigos).fillna(codigos[cfg.CATEGORIA_RARA]).astype(int)  # Aplica la codificación
        return out[VARIABLES].to_numpy(dtype=float)  # Matriz en el orden de VARIABLES

    def desde_lime(self, X):  # Matriz de LIME → DataFrame en el espacio original
        d = pd.DataFrame(X, columns=VARIABLES)  # Matriz a DataFrame
        for c in cfg.VARIABLES_CON_INDICADOR:  # Variables con faltante informativo
            d[c] = d[c].where(d[c] >= 0)  # −1 vuelve a ser nulo
        for c in cfg.CATEGORICAS_MODELO:  # Recorre las categóricas
            d[c] = [self.cats[c][int(round(v))] for v in d[c]]  # Código entero → texto
        return d  # DataFrame en el espacio original

    def explicador(self):  # Crea un explicador nuevo (misma semilla → mismas perturbaciones en ambos entornos)
        return LimeTabularExplainer(  # Explicador tabular
            self.X_train,  # LIME calcula cuartiles y frecuencias de train para perturbar
            feature_names=VARIABLES,  # Nombres de las variables
            categorical_features=[VARIABLES.index(c) for c in cfg.CATEGORICAS_MODELO],  # Índices de las categóricas
            categorical_names={VARIABLES.index(c): self.cats[c] for c in cfg.CATEGORICAS_MODELO},  # Nombres de las categorías
            class_names=["Fully Paid", "Charged Off"],  # Nombres de las clases
            discretize_continuous=True,  # Explicaciones por cuartiles ("x > q3")
            random_state=cfg.SEMILLA,  # Semilla de las perturbaciones
        )

    def instancia(self, id_):  # Fila de test (espacio de LIME) de un préstamo
        return self.a_lime(self.test[self.test[cfg.COL_ID] == id_]).ravel()  # Vector de la instancia


def seleccionar_instancias(test, score_sk, score_sp, umbral_sk, umbral_sp):  # Elige instancias mal clasificadas por ambos modelos
    """Devuelve ({'Falso negativo': id, 'Falso positivo': id}, n_falsos_negativos, n_falsos_positivos)."""
    t = test.set_index(cfg.COL_ID)[[cfg.COL_TARGET]].copy()  # Etiquetas de test por id
    t["pred_sk"] = (score_sk.reindex(t.index) >= umbral_sk).astype(int)  # Clase predicha por scikit-learn
    t["pred_sp"] = (score_sp.reindex(t.index) >= umbral_sp).astype(int)  # Clase predicha por PySpark
    fn = t[(t[cfg.COL_TARGET] == 1) & (t["pred_sk"] == 0) & (t["pred_sp"] == 0)]  # Falsos negativos comunes
    fp = t[(t[cfg.COL_TARGET] == 0) & (t["pred_sk"] == 1) & (t["pred_sp"] == 1)]  # Falsos positivos comunes
    return {"Falso negativo": int(fn.sample(1, random_state=cfg.SEMILLA).index[0]),  # Un falso negativo al azar
            "Falso positivo": int(fp.sample(1, random_state=cfg.SEMILLA).index[0])}, len(fn), len(fp)  # Un falso positivo y los totales


def mejor_probabilistico(metricas):  # Modelo de mayor AUC entre los que entregan probabilidades
    candidatos = [m for m in cfg.NOMBRES_MODELOS if m != "LinearSVC" and m in metricas]  # LinearSVC no produce probabilidades
    return max(candidatos, key=lambda m: metricas[m]["metricas"]["roc_auc"])  # Nombre del mejor modelo

"""Métricas, umbral de decisión, McNemar, bootstrap pareado y corrección de Holm."""

import numpy as np  # Cálculo numérico vectorizado
from joblib import Parallel, delayed  # Paralelización del bootstrap en varios núcleos
from sklearn.metrics import (  # Métricas de clasificación de scikit-learn
    accuracy_score,  # Exactitud
    confusion_matrix,  # Matriz de confusión
    f1_score,  # F1
    precision_score,  # Precisión
    recall_score,  # Sensibilidad (recall)
    roc_auc_score,  # Área bajo la curva ROC
)
from statsmodels.stats.contingency_tables import mcnemar  # Prueba de McNemar
from statsmodels.stats.multitest import multipletests  # Corrección por comparaciones múltiples (Holm)


# ---------------------------------------------------------------------------
# Métricas y umbral
# ---------------------------------------------------------------------------
def metricas(y, pred, puntuacion):  # Calcula las métricas exigidas por el enunciado
    """Accuracy, Precision, Recall, F1, ROC AUC y matriz de confusión."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()  # Celdas de la matriz de confusión
    return {  # Diccionario con todas las métricas
        "accuracy": float(accuracy_score(y, pred)),  # Proporción de aciertos
        "precision": float(precision_score(y, pred, zero_division=0)),  # TP / (TP + FP)
        "recall": float(recall_score(y, pred, zero_division=0)),  # TP / (TP + FN)
        "f1": float(f1_score(y, pred, zero_division=0)),  # Media armónica de precisión y recall
        "roc_auc": float(roc_auc_score(y, puntuacion)),  # AUC calculado con la puntuación continua
        "matriz_confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],  # [[TN, FP], [FN, TP]]
    }


def umbral_optimo_f1(y, puntuacion, n_candidatos=199):  # Umbral que maximiza F1 (se usa SOLO con datos de train)
    """Evalúa como candidatos los cuantiles 0.5 %…99.5 % de la puntuación y devuelve (umbral, f1)."""
    y = np.asarray(y).astype(bool)  # Etiquetas como booleanos
    puntuacion = np.asarray(puntuacion, dtype=float)  # Puntuaciones como flotantes
    candidatos = np.unique(np.quantile(puntuacion, np.linspace(0.005, 0.995, n_candidatos)))  # Umbrales candidatos
    total_pos = y.sum()  # Número total de positivos
    mejor_u, mejor_f1 = candidatos[0], -1.0  # Inicializa el mejor resultado
    for u in candidatos:  # Recorre cada umbral candidato
        pred = puntuacion >= u  # Predicción positiva si la puntuación supera el umbral
        tp = np.sum(pred & y)  # Verdaderos positivos
        f1 = 2 * tp / (pred.sum() + total_pos) if (pred.sum() + total_pos) > 0 else 0.0  # F1 = 2TP / (2TP + FP + FN)
        if f1 > mejor_f1:  # Si mejora el mejor F1 encontrado...
            mejor_u, mejor_f1 = u, f1  # ...se actualiza
    return float(mejor_u), float(mejor_f1)  # Devuelve el umbral y su F1 en train


# ---------------------------------------------------------------------------
# Holm
# ---------------------------------------------------------------------------
def holm(p_valores, alfa=0.05):  # Corrección de Holm dentro de una familia de comparaciones
    """Devuelve (p_ajustados, rechaza_H0)."""
    rechaza, p_aj, _, _ = multipletests(p_valores, alpha=alfa, method="holm")  # Aplica Holm-Bonferroni
    return p_aj, rechaza  # Valores p ajustados y decisión


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------
def prueba_mcnemar(y, pred_1, pred_2):  # Compara los errores de dos clasificadores sobre las mismas observaciones
    """Usa χ² con corrección de continuidad; versión exacta (binomial) si b + c < 25."""
    y = np.asarray(y)  # Etiquetas reales
    acierto_1 = np.asarray(pred_1) == y  # Aciertos del modelo 1
    acierto_2 = np.asarray(pred_2) == y  # Aciertos del modelo 2
    a = int(np.sum(acierto_1 & acierto_2))  # Ambos aciertan
    b = int(np.sum(acierto_1 & ~acierto_2))  # Acierta 1 y falla 2
    c = int(np.sum(~acierto_1 & acierto_2))  # Falla 1 y acierta 2
    d = int(np.sum(~acierto_1 & ~acierto_2))  # Ambos fallan
    exacta = (b + c) < 25  # Criterio del enunciado para usar la versión exacta
    res = mcnemar([[a, b], [c, d]], exact=exacta, correction=True)  # Ejecuta la prueba
    return {"a": a, "b": b, "c": c, "d": d, "estadistico": float(res.statistic),  # Tabla 2×2 y estadístico
            "p_valor": float(res.pvalue), "exacta": exacta}  # Valor p y tipo de prueba


# ---------------------------------------------------------------------------
# Bootstrap pareado con pesos multinomiales (equivalente a remuestrear índices)
# ---------------------------------------------------------------------------
class _MetricasPonderadas:  # Precalcula el orden de las puntuaciones para evaluar AUC y AUC-PR en O(n) por remuestreo
    """Remuestrear con reemplazo equivale a ponderar cada observación por el número de veces que sale."""

    def __init__(self, y, puntuacion, umbral):  # Recibe etiquetas, puntuaciones y umbral de decisión
        self.y = np.asarray(y, dtype=float)  # Etiquetas como flotantes
        valores, grupo = np.unique(np.asarray(puntuacion, dtype=float), return_inverse=True)  # Grupos de puntuaciones iguales (orden ascendente)
        self.grupo = grupo  # Grupo (valor distinto) al que pertenece cada observación
        self.n_grupos = len(valores)  # Número de valores distintos
        self.pred = (np.asarray(puntuacion) >= umbral).astype(float)  # Predicción binaria con el umbral fijo

    def calcular(self, w):  # Calcula AUC, AUC-PR y F1 con pesos w
        pos = np.bincount(self.grupo, weights=w * self.y, minlength=self.n_grupos)  # Peso de positivos por grupo
        neg = np.bincount(self.grupo, weights=w * (1 - self.y), minlength=self.n_grupos)  # Peso de negativos por grupo
        P, N = pos.sum(), neg.sum()  # Totales ponderados de positivos y negativos
        neg_menores = np.cumsum(neg) - neg  # Peso de negativos con puntuación estrictamente menor
        auc = np.sum(pos * (neg_menores + 0.5 * neg)) / (P * N)  # AUC ponderado (empates cuentan 0.5)
        tp = np.cumsum(pos[::-1])  # Verdaderos positivos acumulados desde el umbral más alto
        fp = np.cumsum(neg[::-1])  # Falsos positivos acumulados desde el umbral más alto
        precision = tp / np.maximum(tp + fp, 1e-300)  # Precisión en cada umbral
        recall = tp / P  # Recall en cada umbral
        ap = np.sum(np.diff(np.concatenate([[0.0], recall])) * precision)  # AUC-PR (average precision, igual que sklearn)
        tp_f = np.sum(w * self.pred * self.y)  # TP ponderados con el umbral fijo
        f1 = 2 * tp_f / (np.sum(w * self.pred) + P)  # F1 = 2TP / (predichos positivos + positivos reales)
        return np.array([auc, ap, f1])  # Devuelve las tres métricas


def _bloque_bootstrap(m1, m2, n, semillas):  # Ejecuta un bloque de remuestreos (una tarea de joblib)
    difs = []  # Lista de diferencias por remuestreo
    for s in semillas:  # Recorre las semillas asignadas al bloque
        rng = np.random.default_rng(s)  # Generador independiente y reproducible
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)  # Conteo de veces que sale cada índice (mismos índices para ambos modelos)
        difs.append(m1.calcular(w) - m2.calcular(w))  # Diferencia modelo 1 − modelo 2
    return difs  # Devuelve las diferencias del bloque


def bootstrap_pareado(y, punt_1, punt_2, umbral_1, umbral_2, B=2000, semilla=42, n_jobs=-1):  # Bootstrap pareado
    """Devuelve media e IC percentil 95 % de ΔAUC, ΔAUC-PR y ΔF1 (modelo 1 − modelo 2)."""
    n = len(y)  # Número de observaciones de test
    m1 = _MetricasPonderadas(y, punt_1, umbral_1)  # Precálculo del modelo 1
    m2 = _MetricasPonderadas(y, punt_2, umbral_2)  # Precálculo del modelo 2
    semillas = np.random.SeedSequence(semilla).generate_state(B)  # B semillas derivadas de la semilla fija
    bloques = np.array_split(semillas, 24)  # Divide el trabajo en 24 bloques
    res = Parallel(n_jobs=n_jobs)(delayed(_bloque_bootstrap)(m1, m2, n, b) for b in bloques)  # Ejecuta en paralelo
    difs = np.array([d for bloque in res for d in bloque])  # Matriz B × 3 de diferencias
    salida = {}  # Diccionario de resultados
    for j, nombre in enumerate(["auc", "auc_pr", "f1"]):  # Recorre las tres métricas
        li, ls = np.percentile(difs[:, j], [2.5, 97.5])  # Intervalo percentil del 95 %
        salida[f"delta_{nombre}"] = float(difs[:, j].mean())  # Diferencia media
        salida[f"delta_{nombre}_ic_inf"] = float(li)  # Límite inferior
        salida[f"delta_{nombre}_ic_sup"] = float(ls)  # Límite superior
        salida[f"delta_{nombre}_difiere"] = bool(li > 0 or ls < 0)  # Evidencia de diferencia si el IC excluye el 0
        p_boot = 2 * min(np.mean(difs[:, j] <= 0), np.mean(difs[:, j] >= 0))  # Valor p bootstrap bilateral (proporción al otro lado del 0)
        salida[f"delta_{nombre}_p"] = float(min(1.0, max(p_boot, 1.0 / len(difs))))  # Acotado entre 1/B y 1 (resolución del bootstrap)
    return salida  # Devuelve el resumen

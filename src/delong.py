"""Prueba de DeLong para AUC correlacionados.

Versión rápida O(n log n) de Sun y Xu (2014), adaptada de la implementación de
referencia pública de Yandex Data School (``fast_deLong``, repositorio
``yandexdataschool/roc_comparison``, licencia MIT) y reescrita/comentada para
este proyecto. Incluye la versión directa O(n²) de DeLong et al. (1988) para
validar la rápida sobre conjuntos pequeños.
"""

import numpy as np  # Cálculo numérico vectorizado
from scipy import stats  # Distribución normal para valores p e intervalos


def _rangos_medios(x):  # Rangos con empates promediados
    """Rangos (1..n) asignando el rango medio a los empates."""
    return stats.rankdata(x, method="average")  # scipy implementa el mismo cálculo en C


def delong_rapido(puntuaciones, n_positivos):  # Núcleo del algoritmo de Sun y Xu (2014)
    """
    puntuaciones: matriz (k, n) con las puntuaciones de k modelos, ordenadas de modo que
    las primeras ``n_positivos`` columnas son la clase positiva.
    Devuelve (aucs, covarianza) con la matriz de covarianza k×k de los AUC.
    """
    m = n_positivos  # Número de positivos
    n = puntuaciones.shape[1] - m  # Número de negativos
    k = puntuaciones.shape[0]  # Número de modelos comparados
    pos = puntuaciones[:, :m]  # Puntuaciones de los positivos
    neg = puntuaciones[:, m:]  # Puntuaciones de los negativos
    tx = np.empty((k, m))  # Rangos de cada positivo dentro de los positivos
    ty = np.empty((k, n))  # Rangos de cada negativo dentro de los negativos
    tz = np.empty((k, m + n))  # Rangos de cada observación dentro de todas
    for r in range(k):  # Recorre cada modelo
        tx[r] = _rangos_medios(pos[r])  # Rangos internos de positivos
        ty[r] = _rangos_medios(neg[r])  # Rangos internos de negativos
        tz[r] = _rangos_medios(puntuaciones[r])  # Rangos globales
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n  # AUC = estadístico de Mann-Whitney normalizado
    v01 = (tz[:, :m] - tx) / n  # Componentes estructurales V10 de cada positivo
    v10 = 1.0 - (tz[:, m:] - ty) / m  # Componentes estructurales V01 de cada negativo
    sx = np.atleast_2d(np.cov(v01))  # Covarianza entre modelos de las componentes de positivos
    sy = np.atleast_2d(np.cov(v10))  # Covarianza entre modelos de las componentes de negativos
    covarianza = sx / m + sy / n  # Covarianza de los AUC (DeLong et al., 1988)
    return aucs, covarianza  # Devuelve AUC y matriz de covarianza


def delong_directo(y, puntuaciones):  # Versión O(n²) de DeLong et al. (1988), solo para validar
    """Calcula (aucs, covarianza) comparando explícitamente cada par positivo-negativo."""
    y = np.asarray(y)  # Asegura arreglo numpy
    puntuaciones = np.atleast_2d(np.asarray(puntuaciones, dtype=float))  # Matriz (k, n)
    pos = puntuaciones[:, y == 1]  # Puntuaciones de positivos
    neg = puntuaciones[:, y == 0]  # Puntuaciones de negativos
    m, n = pos.shape[1], neg.shape[1]  # Tamaños de cada clase
    k = puntuaciones.shape[0]  # Número de modelos
    v10 = np.empty((k, m))  # Componente de cada positivo
    v01 = np.empty((k, n))  # Componente de cada negativo
    aucs = np.empty(k)  # AUC de cada modelo
    for r in range(k):  # Recorre cada modelo
        comp = (pos[r][:, None] > neg[r][None, :]).astype(float)  # 1 si el positivo supera al negativo
        comp += 0.5 * (pos[r][:, None] == neg[r][None, :])  # 0.5 en caso de empate
        aucs[r] = comp.mean()  # AUC = proporción de pares bien ordenados
        v10[r] = comp.mean(axis=1)  # Promedio por positivo
        v01[r] = comp.mean(axis=0)  # Promedio por negativo
    covarianza = np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n  # Covarianza de los AUC
    return aucs, covarianza  # Devuelve AUC y covarianza


def _ordenar_por_clase(y, puntuaciones):  # Reordena columnas: primero positivos, luego negativos
    """Prepara la entrada para delong_rapido."""
    y = np.asarray(y).astype(int)  # Etiquetas como enteros
    orden = np.argsort(-y, kind="mergesort")  # Positivos (1) primero, manteniendo el orden relativo
    return np.atleast_2d(np.asarray(puntuaciones, dtype=float))[:, orden], int(y.sum())  # Matriz reordenada y nº de positivos


def ic_auc(y, puntuacion, alfa=0.05):  # AUC con intervalo de confianza de DeLong
    """Devuelve (auc, li, ls) usando el error estándar de DeLong."""
    mat, m = _ordenar_por_clase(y, puntuacion)  # Ordena por clase
    aucs, cov = delong_rapido(mat, m)  # Calcula AUC y varianza
    ee = np.sqrt(cov[0, 0])  # Error estándar del AUC
    z = stats.norm.ppf(1 - alfa / 2)  # Cuantil normal (1.96 para 95 %)
    return aucs[0], aucs[0] - z * ee, aucs[0] + z * ee  # AUC y límites del intervalo


def prueba_delong(y, puntuacion_1, puntuacion_2, alfa=0.05):  # Prueba de DeLong para dos modelos pareados
    """H0: AUC1 = AUC2. Devuelve un diccionario con AUC, IC, ΔAUC, z y valor p bilateral."""
    mat, m = _ordenar_por_clase(y, np.vstack([puntuacion_1, puntuacion_2]))  # Ambos modelos en una matriz
    aucs, cov = delong_rapido(mat, m)  # AUC y matriz de covarianza 2×2
    z_crit = stats.norm.ppf(1 - alfa / 2)  # Cuantil normal para el IC
    delta = aucs[0] - aucs[1]  # Diferencia de AUC
    var_delta = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]  # Var(A−B) = VarA + VarB − 2Cov(A,B)
    ee_delta = np.sqrt(max(var_delta, 1e-300))  # Error estándar de la diferencia (evita división entre 0)
    z = delta / ee_delta  # Estadístico z
    p = 2 * stats.norm.sf(abs(z))  # Valor p bilateral
    ee1, ee2 = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])  # Errores estándar individuales
    return {  # Resultado de la comparación
        "auc_1": aucs[0], "ic_1_inf": aucs[0] - z_crit * ee1, "ic_1_sup": aucs[0] + z_crit * ee1,  # AUC e IC del modelo 1
        "auc_2": aucs[1], "ic_2_inf": aucs[1] - z_crit * ee2, "ic_2_sup": aucs[1] + z_crit * ee2,  # AUC e IC del modelo 2
        "delta_auc": delta, "delta_ic_inf": delta - z_crit * ee_delta, "delta_ic_sup": delta + z_crit * ee_delta,  # ΔAUC e IC
        "z": z, "p_valor": p,  # Estadístico y valor p
    }


def validar_rapido_vs_directo(n=600, k=3, semilla=0):  # Valida la versión rápida frente a la directa
    """Genera datos sintéticos con empates y verifica que ambas versiones coincidan."""
    rng = np.random.default_rng(semilla)  # Generador aleatorio reproducible
    y = rng.integers(0, 2, n)  # Etiquetas aleatorias
    base = y + rng.normal(0, 1.2, n)  # Señal común con ruido
    punt = np.vstack([np.round(base + rng.normal(0, s, n), 1) for s in np.linspace(0.3, 1.5, k)])  # k modelos con empates (redondeo)
    auc_d, cov_d = delong_directo(y, punt)  # Versión directa
    mat, m = _ordenar_por_clase(y, punt)  # Ordena para la versión rápida
    auc_r, cov_r = delong_rapido(mat, m)  # Versión rápida
    return {  # Diferencias máximas absolutas entre ambas versiones
        "max_dif_auc": float(np.max(np.abs(auc_d - auc_r))),  # Diferencia en AUC
        "max_dif_cov": float(np.max(np.abs(cov_d - cov_r))),  # Diferencia en covarianza
        "auc_directo": auc_d, "auc_rapido": auc_r,  # AUC de ambas versiones
        "cov_directo": cov_d, "cov_rapido": cov_r,  # Covarianzas de ambas versiones
    }

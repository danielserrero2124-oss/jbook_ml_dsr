# Proyecto Integrador de Aprendizaje Automático

**Predicción de default en préstamos de Lending Club: scikit-learn vs. PySpark**

Este libro documenta la construcción y comparación de seis modelos de clasificación
supervisada (regresión logística, árbol de decisión, bosque aleatorio, *gradient boosting*,
SVM lineal y Naive Bayes) para predecir si un préstamo de Lending Club termina en *default*
(`Charged Off` = 1) o se paga completamente (`Fully Paid` = 0). Cada modelo se implementa
dos veces, con **scikit-learn** y con **PySpark**, sobre el **dataset completo sin muestreo**
y con **la misma partición 80/20** para que las comparaciones sean pareadas.

## Contenido

| Capítulo | Contenido |
|---|---|
| 1. EDA | Carga completa, análisis uni y bidimensional, multicolinealidad, valores faltantes, resumen ejecutivo |
| 2. Partición y preprocesamiento (scikit-learn) | Partición común estratificada guardada en Parquet; transformadores ajustados solo con train |
| 3. Modelado con scikit-learn | `GridSearchCV` (cv = 3, AUC), métricas, matrices de confusión, tiempos |
| 4. Preprocesamiento y modelado con PySpark | Lectura del CSV en Spark, pipeline, caché, `CrossValidator` (3 pliegues), métricas, tiempos |
| 5. Comparación estadística | DeLong rápido (validado), Holm, McNemar, bootstrap pareado |
| 6. LIME | Explicaciones locales de instancias mal clasificadas en ambos entornos |
| 7. Comparación final | Tablas y gráficos comparativos, escalabilidad y reflexión crítica |

## Resultados principales

| Modelo | AUC scikit-learn | AUC PySpark | Entrenamiento + CV sklearn (s) | Entrenamiento + CV PySpark (s) |
|---|---|---|---|---|
| Gradient boosting | **0,7207** | **0,7208** | 71 | 581 |
| Bosque aleatorio | 0,7153 | 0,7142 | 519 | 2.741 |
| Regresión logística | 0,7141 | 0,7141 | 63 | 27 |
| Árbol de decisión | 0,7043 | 0,7018 | 37 | 26 |
| Naive Bayes | 0,6792 | 0,6795 | 4 | 6 |
| LinearSVC | 0,5293 | 0,6793 | 3.549 | 97 |

* Ambos entornos alcanzan la **misma capacidad predictiva**: el mejor modelo (*gradient
  boosting*) no difiere entre entornos (DeLong, p Holm = 0,82). Solo LinearSVC difiere de forma
  relevante, por diferencias de implementación.
* **Velocidad:** PySpark es más rápido en los modelos lineales y el árbol, y scikit-learn en los
  ensambles. En la regresión logística PySpark supera a scikit-learn a partir de unas 250 mil
  filas; en el bosque, nunca en este equipo.
* Configuración de Spark ajustada y justificada (`local[6]`, 6 particiones, 9g): las 400
  particiones del enunciado resultaron ≈ 9 veces más lentas en una sola máquina, y con 12 hilos
  el bosque de 100 árboles de profundidad 15 agotaba la memoria.

## Reproducibilidad

* Semilla global: 42. Partición 80/20 estratificada (`data/particion_train_test.parquet`).
* Código reutilizable en `src/` y código fuente de cada notebook en `fuentes/`;
  **cada línea de código está comentada**.
* Hardware: AMD Ryzen (6 núcleos físicos / 12 lógicos), 14 GB de RAM, Windows 11,
  Python 3.9, scikit-learn 1.3, PySpark 3.5.3 (modo local), Java 21.

## Referencias

* DeLong, E. R., DeLong, D. M. y Clarke-Pearson, D. L. (1988). *Biometrics*, 44(3), 837–845.
* Dietterich, T. G. (1998). *Neural Computation*, 10(7), 1895–1923.
* Efron, B. y Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
* McNemar, Q. (1947). *Psychometrika*, 12(2), 153–157.
* Sun, X. y Xu, W. (2014). *IEEE Signal Processing Letters*, 21(11), 1389–1393.
* Ribeiro, M. T., Singh, S. y Guestrin, C. (2016). "Why should I trust you?" *KDD*.

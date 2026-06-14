# Modelo Integrado de Riesgo de Quiebra y Valoración Empresarial Ajustada por Riesgo

Repositorio del trabajo de grado para la **Maestría en Ciencias de los Datos y la Analítica** de la **Universidad EAFIT**. El proyecto integra un modelo de *Machine Learning* para estimar la probabilidad de quiebra empresarial con una valoración por flujos de caja descontados (DCF) bajo simulación de Monte Carlo, produciendo una valoración ajustada por riesgo diferenciada empresa por empresa.

**Autora:** Sara Elizabeth Martínez Osorio
**Asesora:** Paula María Almonacid Hurtado
**Universidad EAFIT — Escuela de Ciencias Aplicadas e Ingeniería — Medellín, 2026**

---

## Descripción del proyecto

La predicción de quiebra y la valoración empresarial se han desarrollado tradicionalmente como campos separados: los modelos predictivos no integran sus probabilidades en la valoración, y los métodos de valoración no incorporan el riesgo de quiebra específico de cada empresa. Este trabajo propone un marco integrado que combina ambas dimensiones en dos etapas complementarias:

1. **Riesgo de quiebra (Machine Learning).** Un modelo supervisado estima la probabilidad de quiebra a partir de información financiera histórica observada, con calibración isotónica fuera de muestra para que las probabilidades reflejen la prevalencia real.
2. **Valoración ajustada por riesgo (simulación financiera).** Una simulación de Monte Carlo sobre un modelo DCF estocástico genera la distribución del valor de cada empresa, que se ajusta por la probabilidad calibrada de no quiebra:

   ```
   V_ajustado = (1 − P_quiebra) × V_DCF_MonteCarlo
   ```

Es importante precisar que el trabajo **no desarrolla dos modelos de aprendizaje automático**: el riesgo de quiebra se estima mediante ML sobre datos reales observados, mientras que la valoración se construye mediante simulación financiera. El aporte central es la **integración** de ambos enfoques.

---

## Datos

Se utiliza el **US Companies Bankruptcy Prediction Dataset** (Lombardo et al., 2022), de acceso abierto en la revista *Future Internet* (MDPI), cargado directamente desde su repositorio público:

```
https://raw.githubusercontent.com/sowide/bankruptcy_dataset/main/american_bankruptcy_dataset.csv
```

- **78.682 observaciones** (empresa-año) de **8.971 empresas** estadounidenses cotizadas, período **1999–2018**.
- **18 variables financieras originales** de los estados financieros anuales.
- Variable objetivo binaria: `alive` (activa) / `failed` (quiebra).
- Desbalance de clases: **6,63 %** de eventos de quiebra (razón ≈ 1:14).

> **Nota de unidades:** las cifras monetarias del dataset están expresadas en **miles de USD**. El notebook las convierte a millones de USD en la etapa de valoración para reportar en la escala estándar de finanzas corporativas.

A partir de las 18 variables originales se construyen **52 variables adicionales** (17 ratios financieros —incluidos los scores de Altman y Ohlson— y 35 variables temporales: rezagos, cambios interanuales y medias móviles de tres años), para un total de **70 variables predictoras**.

---

## Metodología

### Prevención de fuga de información (*data leakage*)

El diseño experimental incorpora controles explícitos en todas las etapas:

- **Partición por empresa** (no por observación), con solapamiento verificado igual a cero entre conjuntos: train 70 % / validación 15 % / prueba 15 %.
- **Winsorización** y **escalamiento** ajustados únicamente sobre el conjunto de entrenamiento.
- **Calibración isotónica** estimada fuera de muestra, solo sobre validación.
- **Umbral de clasificación** optimizado sobre validación y aplicado sin modificación a prueba.
- **Validación cruzada Walk-Forward** (5 folds) que preserva el orden cronológico, evitando *look-ahead bias*.

### Modelos evaluados

Se comparan benchmarks clásicos (Altman Z-Score, Ohlson O-Score) frente a 11 modelos de Machine Learning: modelos lineales (Regresión Logística, LinearSVC), ensambles estándar (Random Forest, Gradient Boosting, XGBoost, LightGBM), LightGBM con optimización bayesiana de hiperparámetros (Optuna), modelos especializados en desbalance (BalancedRandomForest, BalancedBagging, EasyEnsemble, RUSBoost) y un Stacking Ensemble.

### Selección del modelo final

El modelo se selecciona por **AUC-PR** (más informativo que AUC-ROC bajo desbalance extremo y coherente con la métrica objetivo de Optuna). Se aplican pruebas de significancia estadística (**test de DeLong** para AUC-ROC y **bootstrap pareado** para AUC-PR) para evitar afirmaciones sobre diferencias que no son significativas. El modelo final es **LightGBM-Optuna**, elegido por parsimonia ante un empate estadístico con el Stacking, con la ventaja adicional de permitir interpretabilidad SHAP directa.

### Simulación de Monte Carlo

- **500 simulaciones por empresa** sobre las 12.009 empresas-año del conjunto de prueba.
- FCFF inicial individualizado por empresa (con jerarquía de *fallbacks*).
- Parámetros estocásticos: WACC ~ N(10 %, 2 %) truncada [4 %, 25 %]; g ~ N(2,5 %, 1,5 %) truncada [0,1 %, 8 %].
- **Truncamiento condicional** g < WACC − δ (δ = 1,5 %) en cada iteración, para acotar el denominador del modelo de Gordon. WACC y g se muestrean de forma independiente (supuesto declarado).
- VaR calculado **por empresa** (percentil 5 de sus 500 simulaciones) y reportado como promedio de los VaR individuales.

---

## Estructura del repositorio

```
.
├── AS-Bankruptcy_V9_corregido.ipynb   # Notebook principal (pipeline completo)
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias
└── outputs/                           # Generado al ejecutar
    ├── models.pkl                     # Modelos entrenados y calibrados
    └── results.pkl                    # Métricas, datos procesados y resultados
```

El notebook está organizado en doce partes:

| Parte | Contenido |
|-------|-----------|
| I | Setup, carga de datos y pipeline base (partición, ingeniería de características) |
| II | Funciones de evaluación y validación Walk-Forward |
| III | Benchmarks clásicos (Altman, Ohlson) |
| IV | Entrenamiento de los 11 modelos de Machine Learning |
| V | Calibración isotónica del top-3 de modelos |
| VI / VI-B | Stacking Ensemble y su calibración |
| VII / VII-B | Tabla maestra de resultados y tests de significancia (DeLong, bootstrap) |
| VIII | Dashboard visual (curvas ROC/PR, estabilidad temporal) |
| IX | Interpretabilidad SHAP |
| X | Resumen ejecutivo |
| XI | Simulación Monte Carlo y modelo integrado riesgo-valor |
| XII | Guardado de modelos, resultados y anexos |

---

## Requisitos e instalación

Requiere **Python 3.10+**. Las dependencias principales:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
optuna
shap
scipy
```

Instalación:

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>
pip install -r requirements.txt
```

---

## Ejecución

El notebook carga el dataset automáticamente desde la URL pública, por lo que solo se requiere conexión a internet. Para reproducir el análisis completo:

```bash
jupyter notebook AS-Bankruptcy_V9_corregido.ipynb
```

Se recomienda ejecutar las celdas en orden. La semilla aleatoria está fijada (`RANDOM_STATE = 42`) para garantizar reproducibilidad. El entrenamiento completo (incluyendo validación Walk-Forward de todos los modelos) puede tardar varios minutos según el equipo.

---

## Resultados principales

| Métrica | Valor |
|---------|-------|
| Modelo final | LightGBM-Optuna |
| AUC-ROC (prueba) | 0,7228 |
| AUC-PR (prueba) | 0,1438 |
| Recall (prueba) | 0,849 (568 de 669 quiebras) |
| Mejora de AUC-ROC sobre Altman Z-Score | +0,1471 (≈ 26 %) |
| Probabilidad media calibrada de quiebra | 7,21 % (prevalencia real: 5,57 %) |
| Descuento medio por riesgo en la valoración | 5,40 % |

Las variables más influyentes en la probabilidad de quiebra (análisis SHAP) corresponden a indicadores de endeudamiento (LTD_TA, Total_LT_debt), rentabilidad acumulada (Retained_Earnings, RE_TA), eficiencia operativa y liquidez.

---

## Limitaciones

- Los datos son exclusivamente de empresas estadounidenses cotizadas (1999–2018), lo que limita la generalización a otros mercados o a empresas privadas.
- La predicción se modela como clasificación binaria; no se estima el tiempo hasta la quiebra (modelos de supervivencia / hazard).
- La valoración asume estado estacionario para el FCFF inicial y crecimiento perpetuo constante (modelo de Gordon); WACC y g se modelan con distribuciones genéricas de mercado, no calibradas individualmente con betas sectoriales.
- El ajuste multiplicativo (1 − P) asume valor de recuperación nulo para el accionista en caso de quiebra; endogeneizar la tasa de recuperación es una línea natural de extensión.

---

## Cita

Si utiliza este trabajo, por favor cite:

> Martínez Osorio, S. E. (2026). *Modelo integrado de riesgo de quiebra mediante Machine Learning y valoración empresarial ajustada por riesgo mediante simulación Monte Carlo* [Trabajo de grado, Maestría en Ciencias de los Datos y la Analítica]. Universidad EAFIT.

### Dataset

> Lombardo, G., Pellegrino, M., Adosoglou, G., Cagnoni, S., Pardalos, P. M., & Poggi, A. (2022). Machine Learning for Bankruptcy Prediction in the American Stock Market: Dataset and Benchmarks. *Future Internet, 14*(8), 244. https://doi.org/10.3390/fi14080244

# Presentación DS — Timely Response (público técnico)

## 1) Contextualización técnica
**Problema**  
Clasificar si una reclamación tendrá **respuesta oportuna** (*“Timely response?” = Yes/No*). El objetivo operacional es **priorizar** casos con riesgo de “No” para reducir costes y mejorar SLAs.

**Datos (esquema típico)**  
Columnas relevantes: `Issue`, `Sub-issue`, `Product`, `Sub-product`, `Company`, `State`, `ZIP code`, `Date received`, `Date sent to company`, `Consumer disputed?`, `Timely response?` (target).
- **Faltantes**: imputación **`State`/`ZIP code`** con reglas:
  - Si hay `ZIP` y falta `State`: asignar el estado correspondiente al ZIP (lookup); 
  - Si hay `State` y falta `ZIP`: asignar el ZIP modal del `State`;
  - Si faltan ambos: `ZIP = -9999`, `State = "NoI"`.
- **Target**: mapeo robusto a binario (`Yes/True/Y`→1, `No/False/N`→0).

**Ingeniería de variables**
- **Texto**: `Issue` + `Sub-issue` → **TF-IDF** (1–2-gramas; `min_df`≥10; `max_features`≈30k).
- **Categóricas**: `Product`, `Sub-product`, `Company`, `State` → **One-Hot** (`handle_unknown="ignore"`).
- **Fecha**: de `Date received` → `year`, `month`, `dow`; si existe `Date sent to company` → `days_to_forward`.
- **Estandarización**: `MaxAbsScaler` para numéricos dispersos.
- **Selección**: `SelectKBest(chi2)` para reducir dimensionalidad de TF-IDF+OHE.

**Arquitectura del pipeline (sklearn)**  
`ColumnTransformer([text, cat, date]) → SelectKBest(chi2) → Estimador`  
Modelos considerados (sin clases, picklables):
- **Logistic Regression** (`class_weight="balanced"`), 
- **LinearSVC** (con calibración o reescalado de `decision_function`),
- **SVC (linear)** con `probability=True`,
- **RandomForest**,
- (Opcionales) **LightGBM/XGBoost** si están disponibles.

**Criterio operativo**  
Los **falsos negativos** (predecir Yes cuando era No) son caros → se optimiza el **umbral** para **maximizar F1 de la clase “No”** en *holdout*.

---

## 2) Enfoque metodológico
1. Limpieza/Imputación de `State/ZIP` y creación del `y` (target).
2. Partición estratificada (*train/test*).
3. Preprocesado reproducible con `Pipeline`/`ColumnTransformer` + `SelectKBest(chi2)`.
4. Búsqueda de hiperparámetros con `GridSearchCV` (CV estratificada, `scoring="balanced_accuracy"`).
5. Barrido de **umbral** en validación para **F1_No** y selección del óptimo.
6. Evaluación en test: **Balanced Accuracy**, **F1_No**, **F1_Yes**, **ROC-AUC**, **PR-AUC**, **matriz de confusión**.
7. Persistencia: `models/trained_model.pkl` + `models/model_config.yaml` (incluye `threshold` y metadatos).
8. Entrega: CLIs (`train-timely`, `predict-timely`) y **app ejecutiva Streamlit** (`app/app_exec.py`).

**Reproducibilidad**  
- Misma versión de **scikit-learn==1.5.1** en entrenamiento y despliegue.
- Import de `src/runtime_transforms` antes de `joblib.load` cuando el pipeline lo requiera.

---

## 3) Resultados y métricas de evaluación
**Qué reportar (ejemplo de plantilla)**  
- Comparativa (`models/model_comparison.csv`): `modelo`, `GS_best_cv_score`, `F1_No@0.5`, `BalAcc@0.5`, `F1_No@opt`, `BalAcc@opt`, `ROC_AUC`, `PR_AUC`, `thr_opt`.
- Matriz de confusión a 0.5 vs umbral óptimo.
- Curvas ROC y PR del mejor modelo.
- Importancias: pesos/coeficientes (LogReg/LinearSVM) o importancias (RF); top tokens/variables más influyentes.

**Generalización y robustez**
- CV estratificada; control de complejidad (C, profundidad, `k` en SelectKBest).
- (Opcional) Split temporal si hay dependencia por fecha/SLA.

---

## 4) Limitaciones y mejoras
- **Datos**: heterogeneidad por `Company/State`; considerar **umbrales por segmento**. Reglas `State/ZIP` pueden enriquecerse.
- **Probabilidades**: calibración (Platt/Isotonic) si se usarán en €.
- **Texto**: probar **embeddings** (p.ej., SBERT) como alternativa a TF-IDF.
- **Aprendizaje sensible a coste** y/o umbrales por segmento.
- **MLOps**: monitor de drift, reentrenos, control de versión de librerías.

---

## 5) Demostraciones prácticas
**Entrenar y guardar artefactos**
```bash
uv run python -m src.train_cli \
  --data data/processed/clean_for_model.csv \
  --out-model models/trained_model.pkl \
  --out-config models/model_config.yaml
```

**Puntuar un CSV procesado**
```bash
uv run python -m src.predict_cli \
  --data data/processed/clean_for_model.csv \
  --model models/trained_model.pkl \
  --config models/model_config.yaml \
  --out models/scored.csv
```

**App ejecutiva (sin cargas externas)**
```bash
uv run streamlit run app/app_exec.py
```
Muestra KPIs de scoring, métricas del modelo y **métricas de negocio** (coste esperado, P90, top compañías y top casos). Descarga `models/scored.csv`.

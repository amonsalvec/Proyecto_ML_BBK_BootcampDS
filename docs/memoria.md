
# Memoria del proyecto — Timely response?

## Objetivo
Predecir si la respuesta de la compañía será oportuna (*Timely response?*) con un modelo interpretable.

## Pipeline
- Texto (Issue/Sub-issue) → TF-IDF (1-2gram, min_df=10, max_features=20k).
- Categóricas (Product, Sub-product, Company, State) → One-Hot.
- Fecha (Date received) → year, month, day, dow + MaxAbsScaler.
- Clasificador: Logistic Regression (class_weight='balanced').

## Umbral operativo
Optimizado por F1 de la clase 'No'.

## Métricas de negocio
- timely_rate, dispute_rate, days_to_forward, avg_cost con escenarios.

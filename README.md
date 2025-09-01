
# PROYECTO FINAL UNIDAD 3 - MACHINE LEARNING

## Integrante:
- Abelardo Monsalve

## 1. Introducción
Este proyecto tiene como objetivo desarrollar un modelo de machine learning desde la adquisición de datos hasta su despliegue 

El archivo .csv corresponde, presumiblemente, a una base de datos de quejas de consumidores probablemente basada en el conjunto de datos de la **Consumer Financial Protection Bureau (CFPB)**, un organismo del gobierno de EE. UU. que supervisa quejas sobre productos y servicios financieros.

## Columnas del archivo CSV
Columna	| Descripción
---|---
Complaint ID | Identificador único de cada queja registrada. Es un número secuencial o un UUID (según la fuente). Sirve como clave primaria.
Product	| Tipo general de producto financiero al que se refiere la queja. Ej: Credit reporting, Mortgage, Debt collection, Credit card, etc.
Sub-product	| Subcategoría más específica del producto. Ej: dentro de Credit reporting, podría ser Credit reporting company used.
Issue |	El problema principal reportado por el consumidor. Ej: Incorrect information on credit report, Loan modification issues, etc.
Sub-issue |	Detalle más específico del problema (puede ser nulo si el consumidor no proporcionó más detalle). Ej: Information belongs to someone else, etc.
State |	Abreviación del estado de EE. UU. desde donde se realizó la queja. Ej: CA (California), TX (Texas), etc.
ZIP code |	Código postal del consumidor (puede estar anonimizado o ausente por privacidad).
Date received |	Fecha en que la CFPB recibió la queja.
Date sent to company |	Fecha en que la CFPB remitió la queja a la empresa señalada.
Company	| Nombre de la empresa a la que se refiere la queja.
Company response |	Tipo de respuesta que dio la empresa. Ej: Closed with explanation, Closed with monetary relief, Untimely response, etc.
Timely response? |	Indica si la empresa respondió en el tiempo estipulado (normalmente 15 días). Valores típicos: Yes, No.
Consumer disputed?	| Indica si el consumidor no estuvo conforme con la respuesta de la empresa (Yes), o no presentó disputa (No).
---

A continuación se presenta una descripción general de la estructura del proyecto y las instrucciones para su uso.

## Estructura del Proyecto

```
nombre_proyecto_final_ML
├── notebooks
│   ├── 01_Fuentes.ipynb: Adquisición de datos y unión de diferentes fuentes.
│   ├── 02_LimpiezaEDA.ipynb: Transformaciones y limpiezas, incluyendo el feature engineering y visualizaciones.
│   ├── 03_Entrenamiento_Evaluacion.ipynb: Entrenamiento de modelos, hiperparametrización y evaluación de modelos.
│   └── ...
├── src
│   ├── preprocessing.py: Código para procesar los datos (escalado, renombrado, descarte de columnas, etc.).
│   ├── training.py: Código para entrenar y guardar el modelo entrenado.
│   ├── evaluation.py: Código para evaluar el modelo y generar métricas de evaluación.
│   └── ...
├── models
│   ├── trained_model.pkl: Modelos entrenados guardados en formato pickle.
│   ├── model_config.yaml: Archivo con la configuración del modelo final.
│   └── ...
├── app
│   ├── app.py: Código para la aplicación web que utiliza el modelo entrenado.
│   ├── requirements.txt: Especifica las dependencias del proyecto.
│   └── ...
├── docs
│   ├── negocio.ppt: Documentación relacionada con el negocio.
│   ├── ds.ppt: Documentación relacionada con la ciencia de datos.
│   ├── memoria.md: Documentación adicional del proyecto.
│   └── ...
└── README.md: Documentación del proyecto.


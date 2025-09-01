# Presentación Negocio — Timely Response

## 1) Contextualización del Problema
**Qué ocurre hoy**  
Cuando una reclamación no recibe una **respuesta oportuna**, aumentan los **costes** (retrabajo, escalados, penalizaciones), el **riesgo reputacional** y la **insatisfacción** del cliente. Además, gestionar un volumen alto de casos con recursos limitados dificulta priorizar a tiempo.

**Qué proponemos**  
Un modelo que **estima la probabilidad** de respuesta oportuna (*Yes/No*) para **priorizar** los casos con mayor riesgo de retraso. Lo traducimos a un **“coste esperado”** por caso para que el negocio decida **dónde actuar primero**.

> Analogía: igual que en un hospital se clasifican o seleccionan pacientes según gravedad, aquí aplicamos la prioridad de reclamaciones según **riesgo** e **impacto económico**.

---

## 2) Explicación del Valor del Modelo
**Cómo ayuda al negocio**
- **Priorización inteligente**: identifica, antes de que ocurra, qué casos tienen más probabilidad de no ser respondidos a tiempo.
- **Reducción de costes**: enfoca al equipo en los casos que más **coste esperado** generan (retrabajo, disputas, penalizaciones de SLA).
- **Mejora de la experiencia**: disminuye demoras y **evita disputas** innecesarias al intervenir antes.
- **Gestión por cliente** (*Company*): permite ver **qué compañías** concentran más riesgo/coste y tomar acciones preventivas (acuerdos de servicio, plantillas, capacidad).

**Cómo funciona en la práctica**
1. El modelo calcula un **score** (0–1) por caso.
2. Convertimos el score en **coste esperado** con 4 parámetros sencillos:  
   - **C0**: coste base por caso (tramitación mínima);  
   - **Pu**: penalización si el caso **no** es oportuno;  
   - **Pd**: penalización por **disputa**;  
   - **Pf**: coste por **día de demora**.  
3. Generamos un **ranking**: primero los casos con mayor coste esperado.  
4. La app ejecutiva muestra **KPIs** y permite descargar un **CSV priorizado** para operar.

---

## 3) Beneficios y Aplicaciones Prácticas
**Beneficios tangibles**
- **Menos retrabajo y escalados** al intervenir preventivamente en casos de alto riesgo.
- **Cumplimiento de SLA**: foco en casos con más impacto antes de vencer plazos.
- **Mejor satisfacción** (menos disputas, respuestas más rápidas).
- **Transparencia y control**: directivos ven KPIs y **pueden ajustar** el umbral de decisión según objetivos.

**Aplicaciones**
- **Operaciones**: lista diaria de **Top casos** por coste esperado (P90 y superiores) para asignación prioritaria.
- **Gestión de clientes (B2B)**: **Top compañías** por coste total esperado; planes de acción específicos.
- **Calidad/Compliance**: seguimiento de **disputas** y tiempos; alertas tempranas.
- **Planificación de capacidad**: dimensionar equipos según la **carga de riesgo** prevista.

---

## 4) Visualización de Resultados (en la app ejecutiva)
La app (Streamlit) muestra pestañas simples y accionables:

- **Visión general**
  - **Timely predicho** (%), **score medio**, distribución de scores, conteo Yes/No.
  - Ajuste del **umbral** para ver el impacto en tiempo real.
- **Métricas de negocio**
  - **Coste esperado medio** y **P90** (umbral de alto coste).
  - **Top compañías** por **coste total esperado** (tabla ordenada: nº casos, timely predicho, coste medio/total).
  - **Top casos** por **coste esperado** (con `Company`, `Issue`, `Product`, `State`, fecha).
- **Métricas de modelo** *(si hay etiqueta real)*
  - **Balanced Accuracy**, **F1 (No/Yes)**, **ROC-AUC**, **PR-AUC** y **Matriz de confusión**.
- **Descargas**
  - **scored.csv** con `score`, `pred` y `expected_cost` para integrar en las operaciones.

> En demos internas, recomendamos mostrar cómo al bajar el **umbral** se reduce el riesgo de **no oportuno**, a costa de revisar más casos (trade-off controlado por negocio).

---

### Cómo usarlo en el día a día
1. Abrir la app ejecutiva: **Timely Response — Ejecutivo**.  
2. Revisar **KPIs** y **P90** del coste esperado.  
3. Ir a **Top casos** y asignar recursos a los de mayor coste esperado.  
4. En **Top compañías**, acordar acciones preventivas y objetivos de mejora.  
5. Descargar **scored.csv** para la operación diaria y seguimiento.

---

### FAQ para directivos
- **¿Podemos ajustar los parámetros de coste?** Sí, desde la app (C0, Pu, Pd, Pf) para análisis de sensibilidad.  
- **¿Qué pasa si cambia la política de SLA?** Se recalibran los parámetros y/o el umbral; el pipeline se reentrena periódicamente.  
- **¿Es explicable?** Sí: se puede activar un informe de **contribuciones** (palabras/claves y campos que más influyen).  
- **¿Qué necesito para producción?** Mantener la misma versión de librerías del entrenamiento, y refrescar el modelo según el ciclo de datos.

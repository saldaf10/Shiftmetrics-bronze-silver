# ShiftMetrics — Roadmap de Visualizacion

**Estado de partida:** 2026-05-26

## 1. Lectura del proyecto

El repo esta bien cubierto en datos, ETL, ML y documentacion tecnica. La capa de visualizacion existe, pero esta repartida entre notebooks de EDA, artefactos graficos del pipeline y la UI de MLflow. No hay aun una capa de producto independiente para consumo ejecutivo o tecnico.

### Evidencia de completitud parcial

- Bronzes, Silver y ML estan documentados y con scripts/notebooks funcionales.
- El modelo cuenta con SHAP, calibration, drift y threshold plots.
- MLflow ya sirve como UI tecnica para experimentos y artefactos.

### Huecos de visualizacion

- No existe dashboard web dedicado.
- No hay pagina unica para seguimiento de KPI, drift, calibracion y explicabilidad.
- Los graficos estan pensados para investigacion, no para lectura gerencial.
- Falta una capa de narrativa visual que conecte Bronze/Silver/Gold con ML.

## 2. Objetivo de la capa de visualizacion

Construir una experiencia unica para responder, rapido y con contexto, estas preguntas:

1. Como esta el flujo de datos de extremo a extremo.
2. Que tan sano esta el modelo en tiempo y por proyecto.
3. Donde estan las alertas de drift, calibracion y threshold.
4. Que explicacion tiene cada prediccion importante.

## 3. Roadmap propuesto

### Fase 0 — Inventario y estandarizacion

- Catalogar todos los graficos ya generados por EDA, SHAP, drift, calibration y threshold.
- Definir nombres estables de artefactos y rutas de salida.
- Identificar que metricas son globales, por proyecto y por sprint.

### Fase 1 — Dashboard tecnico minimo viable

- Crear una app unica para visualizacion tecnica.
- Incluir tabs para:
  - Overview del pipeline.
  - Calidad de datos y cobertura por fuente.
  - Modelo y metricas de validacion.
  - Calibracion y threshold.
  - Drift temporal.
  - SHAP global y local.
- Priorizar lectura en menos de 2 minutos para alguien tecnico.

### Fase 2 — Dashboard ejecutivo

- Resumir el estado del sistema con KPI de alto nivel.
- Mostrar tendencia historica de riesgo por sprint.
- Incluir semaforos de alerta para calibracion y drift.
- Añadir vista por proyecto para comparacion rapida.

### Fase 3 — Gobernanza y trazabilidad

- Vincular cada grafico con su run de MLflow.
- Versionar artefactos visuales con fecha y hash de pipeline.
- Registrar umbrales, dataset split y metrica en el propio panel.

### Fase 4 — Operacion

- Publicar la app con acceso restringido.
- Definir refresco manual o programado.
- Conectar el panel con el pipeline de reentrenamiento y monitoreo.

## 4. Priorizacion para tu trabajo

Como tu alcance es visualizacion, el orden recomendado es:

1. Definir inventario de artefactos.
2. Diseñar el dashboard tecnico minimo viable.
3. Elegir stack de implementacion.
4. Conectar las visualizaciones existentes.
5. Recién despues, armar la capa ejecutiva.

## 5. Criterios de completitud de visualizacion

La parte de visualizacion se puede dar por completa cuando exista:

- Una sola entrada visual al proyecto.
- Un tablero con KPI, drift, calibracion y explicabilidad.
- Artefactos trazables a MLflow.
- Consistencia visual entre etapas del pipeline.
- Documentacion de uso y lectura de metricas.

## 6. Siguiente paso recomendado

Antes de construir nada, conviene decidir el formato del panel:

- Opcion A: Streamlit para rapidez.
- Opcion B: Plotly Dash para control fino.
- Opcion C: Panel interno solo para analisis tecnico.

Mi recomendacion inicial es Streamlit si buscas velocidad de entrega, o Dash si quieres una capa mas formal y escalable.
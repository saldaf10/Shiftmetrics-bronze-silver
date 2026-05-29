# Resumen Ultra-Detallado de Temas y Aplicaciones: Curso Aprendizaje Automático (2026-1)
**Universidad EAFIT** **Profesor:** Pablo A. Saldarriaga  
**Estudiante:** Samuel Andrés Ariza Gómez  

Este documento presenta una reconstrucción exhaustiva, precisa y técnica de cada uno de los temas, metodologías, fórmulas, métricas y aplicaciones prácticas contenidas en las diapositivas de las sesiones del curso.

---

## 📄 Sesión 01 y 02: Fundamentos, Regresión y Clasificación Supervisada

### 1. Aspectos Administrativos del Curso
* **Estructura de Evaluación:**
  * Examen: 25%
  * Seguimiento: 40% (Taller: 20%, Exposición: 20%)
  * Proyecto Integrador: 35%
* **Texto Guía Principal:** Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. Además de James, G. et al. *An Introduction to Statistical Learning*.

### 2. Introducción y Taxonomía del Aprendizaje Automático
* **Taxonomía:** Ubicación precisa de Machine Learning (ML) y Deep Learning (DL) como subcampos de la Inteligencia Artificial (IA).
* **Dicotomía del Aprendizaje:**
  * **Supervisado:** Modelado con datos etiquetados ($X, y$) para predecir variables continuas o categóricas.
  * **No Supervisado:** Descubrimiento de estructuras ocultas y patrones en datos no etiquetados ($X$).
* **Pipeline Estándar de ML:** Recolección de datos, Limpieza/Preprocesamiento, Ingeniería de Atributos (Feature Engineering), Modelado, Evaluación y Despliegue (Deployment).

### 3. Metodología de Validación y Particionamiento
* **División de Datos:** Configuración estricta en tres conjuntos disjuntos para evitar el sesgo de optimismo:
  * **Entrenamiento (Training Set):** Ajuste de los parámetros del modelo.
  * **Validación (Validation Set):** Ajuste de hiperparámetros y selección del mejor modelo.
  * **Prueba (Test Set):** Evaluación final de la capacidad de generalización.

### 4. El Problema de Regresión (Variables Continuas)
* **Objetivo Conceptual:** Estimar la función matemática ideal $f(X)$ tal que $y = f(X) + \epsilon$, donde $\epsilon$ representa el error irreducible.
* **Métricas de Desempeño Aplicadas:**
  * **MAE (Mean Absolute Error):** $\frac{1}{n} \sum |y_i - \hat{y}_i|$. Mide la magnitud promedio del error de forma lineal; robusto ante outliers.
  * **MSE (Mean Squared Error):** $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$. Penaliza de forma cuadrática los errores grandes.
  * **RMSE (Root Mean Squared Error):** $\sqrt{\text{MSE}}$. Devuelve el error a las unidades originales de la variable respuesta $y$.
  * **MAPE (Mean Absolute Percentage Error):** $\frac{1}{n} \sum |\frac{y_i - \hat{y}_i}{y_i}| \times 100$. Mide el error en términos porcentuales relativos.
  * **R-Cuadrado ($R^2$):** Proporción de la varianza total de $y$ explicada por el modelo.

### 5. El Problema de Clasificación (Variables Categóricas)
* **Fundamentos:** Predicción de etiquetas cualitativas mediante asignación de probabilidades $P(Y=1|X)$.
* **Métricas Base (Matriz de Confusión):**
  * **Verdaderos Positivos (TP), Falsos Positivos (FP), Verdaderos Negativos (TN), Falsos Negativos (FN).**
  * **Exactitud (Accuracy):** $\frac{TP + TN}{TP + TN + FP + FN}$. Proporción total de predicciones correctas.
  * **Precisión (Precision):** $\frac{TP}{TP + FP}$. Capacidad de no clasificar como positivo un caso que es negativo (relevante para minimizar falsos positivos).
  * **Sensibilidad / Exhaustividad (Recall / True Positive Rate - TPR):** $\frac{TP}{TP + FN}$. Capacidad del modelo para encontrar todos los casos positivos reales.
  * **F1-Score:** Media armónica entre Precisión y Recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.

### 6. Evaluación Avanzada de Clasificadores
* **Curva ROC (Receiver Operating Characteristic):**
  * Gráfico del **True Positive Rate (TPR / Recall)** en el eje Y frente al **False Free / Positive Rate (FPR)** en el eje X, construido al variar exhaustivamente el umbral de decisión estocástico de 0 a 1.
  * **FPR Fórmula:** $\frac{FP}{TN + FP}$.
  * **AUC-ROC:** Área bajo la curva ROC. Un valor de 0.5 representa un clasificador aleatorio; 1.0 es un clasificador perfecto. Mide la capacidad de ordenamiento y separación de clases del modelo sin importar el umbral.
* **Curva Precisión-Recall (PR Curve):** Gráfico de Precisión (Eje Y) vs. Recall (Eje X). Crucial para conjuntos de datos con desbalanceo severo de clases, donde la curva ROC puede dar una impresión falsamente optimista.
* **Curva Lift y Curva de Ganancia Acumulada (Cumulative Gains):** Herramientas analíticas utilizadas para medir la efectividad de un modelo de clasificación frente a una selección aleatoria, evaluando qué porcentaje del target se captura al ordenar la población por su probabilidad según el modelo (muy común en marketing y scoring).

### 7. Conceptos Teóricos Críticos del Modelado
* **Dilema de Sesgo vs. Varianza (Bias-Variance Trade-off):**
  * **Sesgo (Bias):** Error introducido por supuestos erróneos o simplistas en el algoritmo (causa *Underfitting*).
  * **Varianza (Variance):** Sensibilidad extrema del modelo a pequeñas fluctuaciones en el conjunto de entrenamiento (causa *Overfitting*).
* **Compromiso entre Interpretabilidad y Precisión:** Modelos lineales/árboles simples poseen alta interpretabilidad pero menor flexibilidad analítica; modelos complejos (redes neuronales, ensambles masivos) maximizan la precisión predictiva a costa de convertirse en "cajas negras".
* **Regularización en Árboles de Decisión:**
  * **Pre-pruning (Detener crecimiento):** Restricciones explícitas durante el entrenamiento (`max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`).
  * **Post-pruning (Poda):** Construcción del árbol completo y eliminación posterior de nodos/ramas que aportan una ganancia de impureza inferior a un umbral determinado.

---

## 📄 Sesión 03: Métodos de Ensamble, Series de Tiempo y Desbalanceo de Clases

### 1. Fundamentos de Métodos de Ensamble (Ensemble Learning)
* **Filosofía:** Combinar múltiples "aprendices débiles" (*weak learners*) para construir un "aprendiz fuerte" (*strong learner*) con mayor robustez y menor error.
* **Arquitecturas Principales:**
  * **Bagging (Bootstrap Aggregating):** Entrenamiento en paralelo de múltiples modelos homogéneos sobre subconjuntos de datos extraídos mediante muestreo con reemplazo (Bootstrap). Reduce drásticamente la **varianza**.
  * **Boosting:** Entrenamiento secuencial aditivo de modelos homogéneos. Cada nuevo modelo se enfoca iterativamente en corregir los errores residuales del modelo precedente. Reduce principalmente el **sesgo**.
  * **Stacking (Stacked Generalization):** Entrenamiento en paralelo de modelos heterogéneos (ej. KNN, Regresión Lineal, Árboles) cuyas predicciones se consolidan como atributos de entrada para un meta-modelo final encargado de la ponderación óptima.

### 2. Algoritmos de Ensamble Avanzados
* **Random Forest:**
  * Extensión directa de Bagging aplicada sobre Árboles de Decisión.
  * Añade una doble aleatoriedad: además del muestreo Bootstrap de filas, en cada división de nodo (*split*) se selecciona un subconjunto aleatorio de atributos (*features*).
  * Permite calcular de manera nativa la **Importancia de Variables** (*Feature Importance*) y evaluar mediante el error *Out-Of-Bag* (OOB).
* **Gradient Boosting:**
  * Técnica secuencial donde cada árbol subsiguiente se entrena para predecir los residuos (*pseudo-residuals*) calculados a partir de la derivada de una función de pérdida parametrizable respecto a las predicciones anteriores.
  * Utiliza optimización basada en el Gradiente Descendente en el espacio de funciones.

### 3. Métricas y Validación Específicas para Series de Tiempo
* **El Riesgo de Data Leakage (Fuga de Información):** Demostración analítica de por qué el K-Fold Cross-Validation aleatorio estándar es inválido en series temporales (usar puntos futuros para predecir el pasado destruye la estructura cronológica).
* **Estrategias de Validación Temporal:**
  * **Walk-Forward Validation (Ventana Expansiva):** El conjunto de entrenamiento crece acumulativamente incorporando todos los datos históricos hasta el tiempo $t$, validando en el bloque inmediatamente posterior $t+1$.
  * **Validación con Ventana Deslizante (Rolling Window):** El tamaño del conjunto de entrenamiento permanece constante, desplazándose uniformemente en el tiempo para adaptarse a cambios de régimen (*Concept Drift*).
* **Métricas Especializadas de Series de Tiempo:**
  * **WMAPE (Weighted Mean Absolute Percentage Error):** $\frac{\sum |y_t - \hat{y}_t|}{\sum |y_t|}$. Pondera los errores absolutos por la escala real agregada; evita divisiones por cero indeterminadas e ideal en cadenas de suministro (*Supply Chain*).
  * **SMAPE (Symmetric Mean Absolute Percentage Error):** $\frac{100}{N} \sum \frac{2|y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t|}$. Acotada de manera simétrica entre 0% y 200%.

### 4. El Problema de Desbalanceo de Clases (*Class Imbalance*)
* **Caso de Estudio Aplicado:** Predicción de fallas en redes de energía eléctrica, caracterizado por una distribución severamente sesgada (1.8% de eventos de falla vs. 98.2% de operación normal).
* **Técnicas de Remuestreo (Resampling):**
  * **Undersampling (Submuestreo):**
    * *Random Undersampling:* Eliminación aleatoria de registros de la clase mayoritaria.
    * *Tomek Links:* Identificación y remoción de parejas de puntos colindantes de distinta clase para limpiar y ensanchar la frontera de decisión.
    * *Near Miss / Edited Nearest Neighbor (ENN):* Reglas selectivas basadas en vecinos más cercanos para remover ruido en la clase mayoritaria.
  * **Oversampling (Sobremuestreo):**
    * *Random Oversampling:* Duplicación exacta de instancias de la clase minoritaria (alto riesgo de sobreajuste).
    * *SMOTE (Synthetic Minority Over-sampling Technique):* Generación de instancias sintéticas combinando linealmente atributos de observaciones de la clase minoritaria y sus $k$ vecinos más cercanos.

---

## 📄 Sesión 04: Sistemas de Recomendación

### 1. Arquitectura y Definición del Problema
* **Estructura Tríptica:** Interacciones modeladas formalmente en base a tres entidades: Usuarios ($U$), Ítems ($I$) y la Matriz de Interacciones o Utilidad ($R$).
* **Función de Utilidad:** Estimación del mapeo $u: U \times I \to R$, diseñado para predecir la afinidad latente de un usuario por un ítem no visto.
* **Paradigma de Salida:**
  1. **Predicción de Rating:** Estimar un valor continuo preciso (ej. predecir si la calificación será de 4.2 estrellas).
  2. **Top-N Ranking:** Generar una lista ordenada con los $N$ ítems con mayor probabilidad de consumo para un usuario específico.

### 2. Tipos de Retroalimentación (*Feedback*)
* **Feedback Explícito:** Datos directos y conscientes proporcionados por el usuario. Ejemplos: Calificaciones numéricas (escala 1-5), likes/dislikes, reseñas de texto. Posee alta fidelidad pero padece de extrema dispersión (*sparsity*).
* **Feedback Implícito:** Datos indirectos derivados del comportamiento pasivo monitorizado del usuario. Ejemplos: Clics, historial de compras, reproducciones de video, tiempo de permanencia (*dwell time*). Abundante, pero no discrimina directamente el sesgo negativo (ej. dar clic no garantiza satisfacción).

### 3. Técnicas de Modelado (Filtrado Colaborativo - Enfoque Inicial)
* **Collaborative Filtering:** Filosofía basada en que las preferencias futuras de un usuario pueden predecirse analizando el comportamiento histórico de usuarios con gustos similares.
* **Ventajas:** No requiere metadatos o descriptores explícitos de los ítems o usuarios; detecta patrones sutiles no parametrizados de forma automatizada; promueve la *serendipia*.
* **Desafíos Estructurales:**
  * *Cold Start (Arranque en frío):* Incapacidad absoluta de generar recomendaciones para nuevos usuarios o nuevos ítems debido a la ausencia de interacciones previas.
  * *Sparsity (Dispersión):* Matrices de utilidad donde más del 99% de las celdas están vacías, desestabilizando los cálculos de vecindad.
  * *Popularity Bias (Sesgo de Popularidad):* Tendencia inherente de los algoritmos a recomendar masivamente ítems hiper-populares, sepultando los productos de la "larga cola" (*long tail*).

---

## 📄 Sesión 05: Métricas Avanzadas de Recomendación y Aprendizaje No Supervisado (Clustering)

### 1. Evaluación de Sistemas de Recomendación y Rankings
* **Métricas de Rating:** Aplicación de **RMSE** y **MAE** sobre el subconjunto de test con feedback explícito.
* **Métricas de Clasificación/Relevancia en Top-K:**
  * Establecimiento de un umbral de corte para binarizar la relevancia (ej. Ítem Relevante si $Rating \ge 4$ o si $Watch\text{-}time > 30s$).
  * **Precision@K:** $\frac{\text{Número de ítems recomendados en el Top-K que son relevantes}}{K}$.
  * **Recall@K:** $\frac{\text{Número de ítems recomendados en el Top-K que son relevantes}}{\text{Total de ítems relevantes históricos del usuario}}$.
* **Métricas de Orden y Posición Avanzadas:**
  * **MAP (Mean Average Precision):** Evalúa la precisión media penalizando los ítems relevantes que se colocan abajo en la lista. Se calcula el *Average Precision* (AP) para cada usuario y luego se promedia entre todos los usuarios:
    $$AP = \frac{1}{\text{Ítems Relevantes}} \sum_{k=1}^K \text{Precision}@k \times I(k)$$
    *(Donde $I(k)$ es una función indicadora que vale 1 si el ítem en la posición $k$ es relevante y 0 si no)*.
  * **NDCG (Normalized Discounted Cumulative Gain):** Evalúa la ganancia acumulada de los ítems recomendados reduciendo su valor logarítmicamente según su posición (penaliza fuertemente poner el mejor ítem al final del Top-K).
    $$\text{DCG}@K = \sum_{i=1}^K \frac{\text{Relevancia}_i}{\log_2(i + 1)}$$
    $$\text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$
    *(Donde IDCG es el Ideal DCG, obtenido al ordenar los ítems perfectamente de mayor a menor relevancia)*.

### 2. Dimensiones Multicriterio de Evaluación (Más allá del Accuracy)
* **Cobertura (Coverage):** Porcentaje del catálogo total de ítems que el sistema es capaz de recomendar a al menos un usuario.
* **Diversidad:** Grado de desemejanza entre los ítems incluidos en una misma lista de recomendación (medido usualmente mediante la distancia coseno inversa entre atributos de los ítems).
* **Novedad:** Capacidad del sistema de recomendar ítems que el usuario no conocía o que pertenecen a la cola larga (popularidad inversa).
* **Serendipia:** Medida de qué tan sorpresiva, inesperada y a la vez relevante resulta una recomendación para el usuario.
* **Justicia (Fairness):** Mitigación de sesgos algorítmicos para garantizar exposición equitativa a diferentes creadores/proveedores.
* **Calibración:** Grado en que la distribución de géneros/categorías en las recomendaciones coincide con las proporciones de consumo histórico del usuario.
* **Métricas de Negocio en Línea:** Diseño de Experimentos vía **A/B Testing** para rastrear impactos directos en el producto: CTR (*Click-Through Rate*), Tasas de Conversión, *Engagement* y Métricas de Retención a largo plazo.

### 3. Aprendizaje No Supervisado: Clustering (Agrupamiento)
* **Naturaleza Analítica:** Ausencia total de una variable objetivo o etiqueta guía ($y$). El éxito se define por criterios matemáticos de proximidad geométrica o densidad.
* **Métricas de Distancia Fundamental:**
  * *Distancia Euclidiana ($L_2$):* $d(x, y) = \sqrt{\sum (x_i - y_i)^2}$. Sensible a la escala y la maldición de la dimensionalidad.
  * *Distancia de Manhattan ($L_1$):* $d(x, y) = \sum |x_i - y_i|$. Robusta para espacios vectoriales dispersos.
* **Taxonomía de Estrategias de Clustering:**
  1. **Clustering Jerárquico:** Construcción de una estructura arbórea de agrupamiento (*Dendrograma*). Puede ser *Aglomerativo* (de abajo hacia arriba, uniendo los puntos más cercanos) o *Divisivo* (de arriba hacia abajo). No requiere predefinir el número de clústeres $K$ desde el inicio.
  2. **Clustering No Jerárquico / Particional:** División directa del espacio muestral en un número $K$ predeterminado de grupos disjuntos (ej. K-Means).
  3. **Clustering Basado en Densidad (DBSCAN):** Agrupa puntos basándose en la densidad local de vecindad (parámetros $\epsilon$ y *MinSamples*). Identifica de manera nativa ruido/outliers y clústeres de formas geométricas arbitrarias. Presenta limitaciones si los grupos tienen densidades marcadamente variables.
* **Métricas de Validación de Clústeres:**
  * **Cohesión Interna:** Minimizar la suma de distancias cuadráticas intragrupo.
  * **Separación Externa:** Maximizar la distancia intergrupo entre los centroides o elementos de clústeres vecinos.
  * **Coeficiente de Silueta ($SC$):** Evalúa la calidad del agrupamiento para cada instancia individual:
    $$SC = \frac{b(x) - a(x)}{\max(a(x), b(x))}$$
    * Donde $a(x)$ es la distancia promedio de un punto a los demás miembros de su propio clúster.
    * Donde $b(x)$ es la distancia promedio del punto a los miembros del clúster más cercano.
    * Rango $[-1, 1]$. Valores cercanos a 1 indican una asignación perfecta; valores cercanos a 0 indican solapamiento de fronteras.

---

## 📌 Temas de Vanguardia Asignados para Exposición
Temas específicos avanzados del estado del arte que componen el currículo formal del curso mediante exposiciones obligatorias:
1. **Aprendizaje por Refuerzo:** Modelado de agentes que aprenden a tomar secuencias de decisiones maximizando una recompensa acumulada en un entorno dinámico.
2. **Aprendizaje Semi-supervisado:** Modelos diseñados para entrenar eficientemente combinando pequeñas cantidades de datos etiquetados con grandes volúmenes de datos no etiquetados.
3. **Aprendizaje Federado:** Entrenamiento descentralizado de modelos de ML en múltiples dispositivos locales (edge nodes) protegiendo la privacidad de los datos sin centralizarlos.
4. **MLOps / Deployment:** Metodologías y herramientas operativas para la automatización del ciclo de vida de modelos en producción (CI/CD, monitoreo de data drift).
5. **Incorporación de Restricciones en Modelos:** Técnicas para forzar propiedades matemáticas específicas en los algoritmos, tales como restricciones de monotonía o cotas de valores por razones de negocio.
6. **Explainable AI (XAI):** Técnicas de interpretabilidad post-hoc (ej. SHAP, LIME) para auditar y explicar las predicciones de modelos complejos de caja negra.
7. **Transformers Aplicados a Series de Tiempo:** Adaptación de las arquitecturas de atención (*Self-Attention Mechanism*) para capturar dependencias temporales de largo alcance en datos secuenciales estructurados.

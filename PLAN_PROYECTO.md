# Plan del Proyecto – Plataforma de Análisis de Riesgo Crediticio

## Objetivo del Proyecto

El objetivo de este proyecto es simular el desarrollo de un **sistema de machine learning orientado a producción para la evaluación de riesgo crediticio**.

El sistema predice la probabilidad de que un prestatario incumpla el pago de un préstamo y estima la pérdida financiera esperada.

Este proyecto no se centra únicamente en el modelo de machine learning, sino también en la **infraestructura de ingeniería necesaria para desplegar y operar un modelo de riesgo crediticio**.

---

# Contexto de Negocio

Las instituciones financieras evalúan solicitudes de préstamo estimando el riesgo de que un prestatario no pague su deuda.

Dos métricas clave se utilizan comúnmente:

- **Probabilidad de Incumplimiento (PD)** – probabilidad de que el prestatario incumpla el pago  
- **Pérdida Esperada (EL)** – pérdida financiera estimada en caso de incumplimiento  

La pérdida esperada normalmente se calcula como:


Pérdida Esperada = PD × LGD × EAD


Donde:

- **PD** = Probabilidad de Incumplimiento  
- **LGD** = Pérdida Dada el Incumplimiento  
- **EAD** = Exposición al Momento del Incumplimiento  

---

# Componentes del Sistema

El proyecto está diseñado como una plataforma completa de machine learning que incluye:

### Generación de Datos

Se generan datos crediticios sintéticos para simular conjuntos de datos financieros reales.

### Preprocesamiento de Datos

Limpieza de datos, manejo de valores faltantes y preparación de variables.

### Ingeniería de Características

Transformación de variables originales en variables adecuadas para el modelo.

### Entrenamiento del Modelo

Entrenamiento de un modelo de regresión logística para estimar la probabilidad de incumplimiento.

### Inferencia del Modelo

Pipeline de predicción que calcula:

- Probabilidad de incumplimiento  
- Pérdida esperada  
- Decisión crediticia  

### Despliegue mediante API

Un servicio con **FastAPI** expone el modelo para realizar predicciones en tiempo real.

### Registro en Base de Datos

Las predicciones se almacenan en **PostgreSQL** para simular un sistema de registro en producción.

### Scoring por Lotes

Un pipeline permite evaluar múltiples solicitudes de préstamo de forma simultánea.

---

# Etapas de Desarrollo

### Etapa 1 – Simulación de Datos

Generar conjuntos de datos crediticios sintéticos con propiedades realistas, incluyendo:

- variables categóricas  
- valores faltantes  
- valores atípicos  
- ruido en los datos  

---

### Etapa 2 – Pipeline de Ingeniería de Características

Crear pipelines reutilizables para el preprocesamiento y transformación de datos.

---

### Etapa 3 – Entrenamiento del Modelo

Entrenar y evaluar un modelo de regresión logística para riesgo crediticio.

---

### Etapa 4 – Pipeline de Inferencia

Construir el pipeline de predicción que incluya:

- cálculo de probabilidad de incumplimiento (PD)  
- estimación de pérdida esperada  
- reglas de decisión crediticia  

---

### Etapa 5 – Despliegue de la API

Desplegar el modelo utilizando **FastAPI** para habilitar predicciones en tiempo real.

---

### Etapa 6 – Infraestructura con Docker

Containerizar la aplicación y la base de datos utilizando **Docker y Docker Compose**.

---

### Etapa 7 – Registro de Predicciones

Almacenar los resultados de las predicciones en **PostgreSQL** para monitoreo y auditoría.

---

### Etapa 8 – Scoring Crediticio por Lotes

Implementar un sistema de scoring por lotes para simular la evaluación de grandes volúmenes de solicitudes de préstamo.

---

# Mejoras Futuras

Posibles extensiones del proyecto incluyen:

- monitoreo de modelos  
- reentrenamiento automático  
- seguimiento de experimentos  
- pipelines de CI/CD  
- orquestación de workflows (Airflow o Prefect)  
- registro de modelos (model registry)  

---

# Autor

**Renzo Salazar**  

Machine Learning y Ciencia de Datos
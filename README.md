# Plataforma de Análisis de Riesgo Crediticio

Sistema completo de Machine Learning para predicción de riesgo crediticio, incluyendo entrenamiento del modelo, ingeniería de variables, despliegue de API, scoring batch y almacenamiento de predicciones en base de datos.

Este proyecto simula cómo las instituciones financieras despliegan modelos de riesgo crediticio en entornos de producción.

---

# Descripción del Proyecto

El objetivo de este proyecto es construir un **sistema de Machine Learning con arquitectura similar a producción**, capaz de:

- Entrenar un modelo de riesgo crediticio
- Servir predicciones a través de una API
- Registrar predicciones en una base de datos
- Ejecutar scoring batch para múltiples solicitudes de crédito
- Desplegar el sistema utilizando Docker

El sistema estima dos métricas clave de riesgo crediticio:

- **Probabilidad de Incumplimiento (Probability of Default - PD)**
- **Pérdida Esperada (Expected Loss - EL)**

---

# Arquitectura del Sistema

Cliente / Pipeline Batch  
↓  
Servicio de Predicción (FastAPI)  
↓  
Modelo de Machine Learning (Regresión Logística)  
↓  
Base de Datos PostgreSQL (Registro de Predicciones)

El sistema permite dos tipos de scoring:

1. **Scoring en tiempo real** mediante API
2. **Scoring batch** para múltiples solicitudes de crédito

---

# Funcionalidades

- Entrenamiento de modelo de riesgo crediticio
- Pipeline de ingeniería de variables
- API de predicción con FastAPI
- Despliegue mediante Docker
- Registro de predicciones en PostgreSQL
- Pipeline de scoring batch
- Generación de dataset sintético de crédito

---

# Tecnologías Utilizadas

### Machine Learning
- Python
- Scikit-learn
- NumPy
- Pandas

### Backend
- FastAPI
- Uvicorn

### Infraestructura
- Docker
- Docker Compose

### Base de Datos
- PostgreSQL
- psycopg2

---

# Estructura del Proyecto


credit-risk-analytics-platform/

app/

data/

models/
best_model_logistic_regression.pkl
model_metrics.json

notebooks/
eda.ipynb

reports/

sql/
init_db.sql

src/
api.py
batch_scoring.py
data_simulation.py
database.py
feature_engineering.py
inference.py
predict.py
preprocessing.py
risk_engine.py
schemas.py
train_model.py

docker-compose.yml
Dockerfile
requirements.txt
README.md
README_ES.md
PROJECT_PLAN.md


---

# Cómo Ejecutar el Proyecto

## 1 Clonar el repositorio


git clone https://github.com/renzosalazar13/credit-risk-analytics-platform.git

cd credit-risk-analytics-platform


---

## 2 Iniciar los servicios con Docker


docker-compose up -d


Esto inicia:

- Servicio de API con FastAPI
- Base de datos PostgreSQL

---

## 3 Crear la tabla en la base de datos


docker exec -it credit-risk-db psql -U mluser -d creditrisk


Luego ejecutar:


CREATE TABLE predictions (
id SERIAL PRIMARY KEY,
timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
model_version TEXT,
age INT,
income FLOAT,
probability_default FLOAT,
expected_loss FLOAT,
credit_decision TEXT
);


Salir de PostgreSQL:


\q


---

## 4 Ejecutar scoring batch


python -m src.batch_scoring


Esto simula la evaluación de múltiples solicitudes de crédito y guarda los resultados en PostgreSQL.

---

## 5 Acceder a la documentación de la API

Abrir en el navegador:


http://localhost:8000/docs


Desde esta interfaz se pueden probar las predicciones directamente.

---

# Ejemplo de Predicción

Entrada


{
"age": 40,
"employment_years": 10,
"employment_type": "salaried",
"region": "urban",
"annual_income": 85000,
"current_debt": 20000,
"debt_to_income_ratio": 0.24,
"credit_utilization": 0.35,
"number_of_credit_lines": 4,
"loan_amount": 15000,
"loan_purpose": "car",
"loan_term_months": 36,
"interest_rate": 0.12,
"late_payments_last_12m": 1,
"recent_credit_inquiries": 2,
"account_tenure_months": 48
}


Salida


{
"probability_of_default": 0.1891,
"LGD": 0.45,
"EAD": 15000,
"expected_loss": 1513.07,
"decision": "REVIEW"
}


---

# Scoring Batch

El scoring batch simula procesos reales en instituciones financieras donde miles de solicitudes de crédito son evaluadas automáticamente.

Pasos del pipeline:

1. Cargar el modelo entrenado
2. Cargar o generar datos de entrada
3. Ejecutar predicciones
4. Guardar resultados en PostgreSQL

---

# Mejoras Futuras

Posibles extensiones del proyecto:

- Monitoreo del modelo
- Integración con Feature Store
- Pipeline CI/CD
- Tracking de experimentos con MLflow
- Orquestación de workflows con Airflow

---

# Autor

Renzo Salazar

Machine Learning y Ciencia de Datos

---

# Licencia

Este proyecto fue desarrollado con fines educativos y de portafolio profesional
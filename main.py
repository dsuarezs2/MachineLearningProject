import os
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using Random Forest, Logistic Regression, and Decision Tree.",
    version="1.0.0"
)

# Servir archivos estáticos
os.makedirs('static', exist_ok=True)

# Pydantic model for patient input
class PatientData(BaseModel):
    age: float
    sex: str  # 'Male', 'Female'
    dataset: str  # 'Cleveland', 'Hungary', 'Switzerland', 'VA Long Beach'
    cp: str  # 'typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'
    trestbps: float | None = None
    chol: float | None = None
    fbs: bool | None = None
    restecg: str | None = None
    thalch: float | None = None
    exang: bool | None = None
    oldpeak: float | None = None
    slope: str | None = None
    ca: float | None = None
    thal: str | None = None
    model_name: str = 'rf'  # 'rf', 'lr', 'dt'

# Global variables for models and parameters
models = {}
scaler = None
feature_columns = []
pipeline_params = {}

@app.on_event("startup")
def startup_event():
    global scaler, feature_columns, pipeline_params
    
    # Rutas de los modelos
    rf_path = 'models/rf_model.joblib'
    lr_path = 'models/lr_model.joblib'
    dt_path = 'models/dt_model.joblib'
    scaler_path = 'models/scaler.joblib'
    cols_path = 'models/model_columns.json'
    params_path = 'models/pipeline_params.json'
    
    # Verificar si existen los modelos
    if not (os.path.exists(rf_path) and os.path.exists(scaler_path)):
        print("Modelos no encontrados. Ejecutando entrenamiento automático...")
        import train
        train.main()
        
    # Cargar modelos y parámetros
    try:
        models['rf'] = joblib.load(rf_path)
        models['lr'] = joblib.load(lr_path)
        models['dt'] = joblib.load(dt_path)
        scaler = joblib.load(scaler_path)
        
        with open(cols_path, 'r') as f:
            feature_columns = json.load(f)
            
        with open(params_path, 'r') as f:
            pipeline_params = json.load(f)
            
        print("¡Modelos y parámetros cargados correctamente!")
    except Exception as e:
        print(f"Error al cargar los modelos: {e}")

def preprocess_input(data: dict, feature_columns, pipeline_params):
    # Imputar si es None
    d = {
        'age': data.get('age'),
        'sex': data.get('sex'),
        'dataset': data.get('dataset'),
        'cp': data.get('cp'),
        'trestbps': data.get('trestbps') if data.get('trestbps') is not None else pipeline_params['medians']['trestbps'],
        'chol': data.get('chol') if data.get('chol') is not None else pipeline_params['medians']['chol'],
        'fbs': data.get('fbs') if data.get('fbs') is not None else pipeline_params['modes']['fbs'],
        'restecg': data.get('restecg') if data.get('restecg') is not None else pipeline_params['modes']['restecg'],
        'thalch': data.get('thalch') if data.get('thalch') is not None else pipeline_params['medians']['thalch'],
        'exang': data.get('exang') if data.get('exang') is not None else pipeline_params['modes']['exang'],
        'oldpeak': data.get('oldpeak') if data.get('oldpeak') is not None else pipeline_params['medians']['oldpeak'],
        'slope': data.get('slope') if data.get('slope') is not None else pipeline_params['modes']['slope'],
        'ca': data.get('ca') if data.get('ca') is not None else pipeline_params['medians']['ca'],
        'thal': data.get('thal') if data.get('thal') is not None else pipeline_params['modes']['thal']
    }
    
    # Corregir ceros
    if d['chol'] == 0:
        d['chol'] = pipeline_params['medians']['chol']
    if d['trestbps'] == 0:
        d['trestbps'] = pipeline_params['medians']['trestbps']
        
    # Capping outliers
    if d['trestbps'] > 200: d['trestbps'] = 200.0
    if d['chol'] > 500: d['chol'] = 500.0
    if d['thalch'] < 60: d['thalch'] = 60.0
    if d['oldpeak'] > 5: d['oldpeak'] = 5.0
    if d['oldpeak'] < 0: d['oldpeak'] = 0.0
    
    # Tipos correctos
    d['fbs'] = bool(d['fbs'])
    d['exang'] = bool(d['exang'])
    
    row_df = pd.DataFrame([d])
    
    # Crear DataFrame vacío alineado con las columnas de entrenamiento
    final_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Llenar las columnas numéricas y booleanas directas
    for col in ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'fbs', 'exang']:
        if col in final_df.columns:
            final_df.loc[0, col] = row_df.loc[0, col]
            
    # Llenar variables dummies
    for cat_col in ['sex', 'dataset', 'cp', 'restecg', 'slope', 'thal']:
        val = row_df.loc[0, cat_col]
        dummy_col = f"{cat_col}_{val}"
        if dummy_col in final_df.columns:
            final_df.loc[0, dummy_col] = 1
            
    return final_df

@app.post("/predict")
async def predict(data: PatientData):
    model_name = data.model_name.lower()
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Modelo '{model_name}' no soportado. Elija uno de: rf, lr, dt")
        
    model = models[model_name]
    
    # Convertir Pydantic a dict
    patient_dict = data.dict()
    
    # 1. Preprocesar entrada original
    input_df = preprocess_input(patient_dict, feature_columns, pipeline_params)
    input_scaled = scaler.transform(input_df)
    
    # 2. Predecir
    prob = float(model.predict_proba(input_scaled)[0][1])
    prediction = int(model.predict(input_scaled)[0])
    
    # 3. Calcular contribuciones locales (Feature Occlusion / Perturbation)
    # Definimos los valores de referencia para cada variable en caso de perturbación
    baselines = {
        'age': 54.0,  # edad promedio/mediana aprox.
        'sex': 'Female',  # sexo base
        'dataset': 'Cleveland',  # dataset base
        'cp': 'asymptomatic',  # dolor de pecho base (el más común en enfermos)
        'trestbps': pipeline_params['medians']['trestbps'],
        'chol': pipeline_params['medians']['chol'],
        'fbs': pipeline_params['modes']['fbs'],
        'restecg': pipeline_params['modes']['restecg'],
        'thalch': pipeline_params['medians']['thalch'],
        'exang': pipeline_params['modes']['exang'],
        'oldpeak': pipeline_params['medians']['oldpeak'],
        'slope': pipeline_params['modes']['slope'],
        'ca': pipeline_params['medians']['ca'],
        'thal': pipeline_params['modes']['thal']
    }
    
    contributions = []
    features_to_explain = list(baselines.keys())
    
    for feat in features_to_explain:
        # Copiar paciente y perturbar la variable
        perturbed_patient = patient_dict.copy()
        perturbed_patient[feat] = baselines[feat]
        
        # Preprocesar y predecir para el paciente perturbado
        perturbed_df = preprocess_input(perturbed_patient, feature_columns, pipeline_params)
        perturbed_scaled = scaler.transform(perturbed_df)
        perturbed_prob = float(model.predict_proba(perturbed_scaled)[0][1])
        
        # Contribución: cambio en la probabilidad (original - perturbado)
        # Si original > perturbado, entonces el valor original aumentó el riesgo
        change = prob - perturbed_prob
        
        # Nombres amigables para el frontend
        friendly_names = {
            'age': 'Edad',
            'sex': 'Sexo',
            'dataset': 'Institución Médica',
            'cp': 'Tipo Dolor Pecho',
            'trestbps': 'Presión Arterial',
            'chol': 'Colesterol',
            'fbs': 'Azúcar en Sangre',
            'restecg': 'Electrocardiograma',
            'thalch': 'Frecuencia Cardíaca Máx',
            'exang': 'Angina por Ejercicio',
            'oldpeak': 'Depresión ST (Oldpeak)',
            'slope': 'Pendiente ST (Slope)',
            'ca': 'Vasos Coloreados (Flouroscopía)',
            'thal': 'Talasemia'
        }
        
        # Detalle de lo que tenía y lo que representa la perturbación
        original_val = patient_dict[feat]
        if feat == 'fbs':
            original_val_str = 'Alto (>120 mg/dl)' if original_val else 'Normal (≤120 mg/dl)'
        elif feat == 'exang':
            original_val_str = 'Sí' if original_val else 'No'
        elif feat in ['sex', 'dataset', 'cp', 'restecg', 'slope', 'thal']:
            original_val_str = str(original_val).title()
        else:
            original_val_str = f"{original_val}"
            
        contributions.append({
            'key': feat,
            'name': friendly_names[feat],
            'value': original_val_str,
            'impact': change
        })
        
    # Ordenar contribuciones por magnitud de impacto absoluto
    contributions = sorted(contributions, key=lambda x: abs(x['impact']), reverse=True)
    
    # Determinar el nivel de riesgo
    if prob < 0.35:
        risk_level = "Bajo"
    elif prob < 0.65:
        risk_level = "Moderado"
    else:
        risk_level = "Alto"
        
    return {
        'probability': prob,
        'prediction': prediction,
        'risk_level': risk_level,
        'contributions': contributions,
        'model_used': model_name
    }

# Servir index.html de forma directa
@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

# Servir estáticos
app.mount("/", StaticFiles(directory="static"), name="static")

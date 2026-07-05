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
    description="Predicts heart disease risk using Random Forest, Logistic Regression, and Decision Tree.",
    version="1.0.0"
)

os.makedirs('static', exist_ok=True)

# ─────────────────────────────────────────────
# Pydantic model
# ─────────────────────────────────────────────
class PatientData(BaseModel):
    age: float
    sex: str              # 'Male' | 'Female'
    dataset: str          # 'Cleveland' | 'Hungary' | 'Switzerland' | 'VA Long Beach'
    cp: str               # 'typical angina' | 'atypical angina' | 'non-anginal' | 'asymptomatic'
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
    model_name: str = 'rf'  # 'rf' | 'lr' | 'dt'

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
models: dict = {}
scaler = None
feature_columns: list = []
pipeline_params: dict = {}

FRIENDLY_NAMES = {
    'age':      'Edad',
    'sex':      'Sexo',
    'dataset':  'Institución Médica',
    'cp':       'Tipo Dolor Pecho',
    'trestbps': 'Presión Arterial',
    'chol':     'Colesterol',
    'fbs':      'Azúcar en Sangre',
    'restecg':  'Electrocardiograma',
    'thalch':   'Frecuencia Cardíaca Máx',
    'exang':    'Angina por Ejercicio',
    'oldpeak':  'Depresión ST (Oldpeak)',
    'slope':    'Pendiente ST',
    'ca':       'Vasos Coloreados',
    'thal':     'Talasemia',
}

# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    global scaler, feature_columns, pipeline_params

    if not os.path.exists('models/rf_model.joblib'):
        print("Modelos no encontrados. Entrenando...")
        import train
        train.main()

    models['rf'] = joblib.load('models/rf_model.joblib')
    models['lr'] = joblib.load('models/lr_model.joblib')
    models['dt'] = joblib.load('models/dt_model.joblib')
    scaler        = joblib.load('models/scaler.joblib')

    with open('models/model_columns.json') as f:
        feature_columns = json.load(f)
    with open('models/pipeline_params.json') as f:
        pipeline_params = json.load(f)

    print("¡Modelos y parámetros cargados!")

# ─────────────────────────────────────────────
# Preprocessing helper
# ─────────────────────────────────────────────
def patient_to_row(d: dict) -> dict:
    """Impute, cap, and return a flat dict of raw feature values."""
    med = pipeline_params['medians']
    mod = pipeline_params['modes']

    row = {
        'age':      float(d['age']),
        'sex':      d['sex'],
        'dataset':  d['dataset'],
        'cp':       d['cp'],
        'trestbps': float(d['trestbps']) if d.get('trestbps') is not None else med['trestbps'],
        'chol':     float(d['chol'])     if d.get('chol')     is not None else med['chol'],
        'fbs':      bool(d['fbs'])       if d.get('fbs')      is not None else bool(mod['fbs']),
        'restecg':  d['restecg']         if d.get('restecg')  is not None else mod['restecg'],
        'thalch':   float(d['thalch'])   if d.get('thalch')   is not None else med['thalch'],
        'exang':    bool(d['exang'])     if d.get('exang')    is not None else bool(mod['exang']),
        'oldpeak':  float(d['oldpeak'])  if d.get('oldpeak')  is not None else med['oldpeak'],
        'slope':    d['slope']           if d.get('slope')    is not None else mod['slope'],
        'ca':       float(d['ca'])       if d.get('ca')       is not None else med['ca'],
        'thal':     d['thal']            if d.get('thal')     is not None else mod['thal'],
    }

    # Correct physiologically impossible zeros
    if row['chol'] == 0:    row['chol']    = med['chol']
    if row['trestbps'] == 0: row['trestbps'] = med['trestbps']

    # Capping
    row['trestbps'] = min(row['trestbps'], 200.0)
    row['chol']     = min(row['chol'],     500.0)
    row['thalch']   = max(row['thalch'],    60.0)
    row['oldpeak']  = max(0.0, min(row['oldpeak'], 5.0))

    return row


def row_to_array(row: dict) -> np.ndarray:
    """Encode one patient row into the model's feature vector."""
    # Build a single-row DataFrame aligned to training columns
    # (start with all zeros as floats)
    vec = {col: 0.0 for col in feature_columns}

    # Numeric + bool features that survive directly (no dummy prefix)
    direct_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'fbs', 'exang']
    for col in direct_cols:
        if col in vec:
            vec[col] = float(row[col])

    # Categorical dummies
    for cat_col in ['sex', 'dataset', 'cp', 'restecg', 'slope', 'thal']:
        dummy_key = f"{cat_col}_{row[cat_col]}"
        if dummy_key in vec:
            vec[dummy_key] = 1.0

    return np.array([list(vec.values())], dtype=np.float64)


def friendly_value(feat: str, raw_value) -> str:
    if feat == 'fbs':
        return 'Alto (>120 mg/dl)' if raw_value else 'Normal (≤120 mg/dl)'
    if feat == 'exang':
        return 'Sí' if raw_value else 'No'
    return str(raw_value).title()


# ─────────────────────────────────────────────
# Predict endpoint
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(data: PatientData):
    mname = data.model_name.lower()
    if mname not in models:
        raise HTTPException(400, f"Modelo '{mname}' no soportado. Usa: rf, lr, dt")

    model   = models[mname]
    patient = data.model_dump()

    # 1. Preprocess original patient
    row_orig = patient_to_row(patient)
    X_orig   = scaler.transform(row_to_array(row_orig))
    prob     = float(model.predict_proba(X_orig)[0][1])
    pred     = int(model.predict(X_orig)[0])

    # 2. Risk level
    if prob < 0.35:
        risk_level = "Bajo"
    elif prob < 0.65:
        risk_level = "Moderado"
    else:
        risk_level = "Alto"

    # 3. Feature contributions via perturbation (occlusion)
    baselines = {
        'age':      54.0,
        'sex':      'Female',
        'dataset':  'Cleveland',
        'cp':       'asymptomatic',
        'trestbps': pipeline_params['medians']['trestbps'],
        'chol':     pipeline_params['medians']['chol'],
        'fbs':      pipeline_params['modes']['fbs'],
        'restecg':  pipeline_params['modes']['restecg'],
        'thalch':   pipeline_params['medians']['thalch'],
        'exang':    pipeline_params['modes']['exang'],
        'oldpeak':  pipeline_params['medians']['oldpeak'],
        'slope':    pipeline_params['modes']['slope'],
        'ca':       pipeline_params['medians']['ca'],
        'thal':     pipeline_params['modes']['thal'],
    }

    contributions = []
    for feat, baseline_val in baselines.items():
        perturbed = {**patient, feat: baseline_val}
        row_p     = patient_to_row(perturbed)
        X_p       = scaler.transform(row_to_array(row_p))
        prob_p    = float(model.predict_proba(X_p)[0][1])

        contributions.append({
            'key':    feat,
            'name':   FRIENDLY_NAMES.get(feat, feat),
            'value':  friendly_value(feat, row_orig[feat]),
            'impact': prob - prob_p,   # >0 means this value increased risk
        })

    contributions.sort(key=lambda x: abs(x['impact']), reverse=True)

    return {
        'probability':   prob,
        'prediction':    pred,
        'risk_level':    risk_level,
        'contributions': contributions,
        'model_used':    mname,
    }

# ─────────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────────
@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

app.mount("/", StaticFiles(directory="static"), name="static")

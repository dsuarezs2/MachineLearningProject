import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    print("Iniciando pipeline de entrenamiento...")
    
    # 1. Cargar el dataset
    csv_path = 'heart_disease_uci.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo {csv_path} en el directorio de trabajo.")
        
    df = pd.read_csv(csv_path)
    print(f"Dataset original cargado: {df.shape[0]} filas, {df.shape[1]} columnas.")
    
    # 2. Crear variable objetivo binaria
    df['heart_disease'] = (df['num'] > 0).astype(int)
    
    # 3. Eliminar duplicados
    df = df.drop_duplicates()
    
    # 4. Eliminar columna id y num, age_binned si existen
    cols_to_drop = ['id', 'num', 'age_binned']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # 5. Calcular y guardar parámetros de limpieza/imputación antes de hacer cualquier cambio
    # Esto es necesario para que el API de inferencia impute de la misma forma.
    numerical_cols = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    categorical_cols = ['fbs', 'exang', 'restecg', 'slope', 'thal', 'sex', 'dataset', 'cp']
    
    # Medias/Medianas y Modas
    pipeline_params = {
        'medians': {},
        'modes': {}
    }
    
    # Imputar variables numéricas en el dataset de entrenamiento
    for col in ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']:
        val = float(df[col].median())
        pipeline_params['medians'][col] = val
        df[col].fillna(val, inplace=True)
        
    # Imputar variables categóricas en el dataset de entrenamiento
    for col in ['fbs', 'exang', 'restecg', 'slope', 'thal']:
        # Convertir a booleanos fbs y exang si tienen valores TRUE/FALSE o similares
        if col in ['fbs', 'exang']:
            # Mapear strings a booleanos
            df[col] = df[col].map({'TRUE': True, 'FALSE': False, True: True, False: False, 'True': True, 'False': False})
            val = bool(df[col].mode()[0])
        else:
            val = str(df[col].mode()[0])
            
        pipeline_params['modes'][col] = val
        df[col].fillna(val, inplace=True)
        
    # 6. Corregir ceros fisiológicos imposibles con las medianas correspondientes
    df.loc[df['chol'] == 0, 'chol'] = pipeline_params['medians']['chol']
    df.loc[df['trestbps'] == 0, 'trestbps'] = pipeline_params['medians']['trestbps']
    
    # 7. Capping de outliers en variables numéricas
    df.loc[df['trestbps'] > 200, 'trestbps'] = 200
    df.loc[df['chol'] > 500, 'chol'] = 500
    df.loc[df['thalch'] < 60, 'thalch'] = 60
    df.loc[df['oldpeak'] > 5, 'oldpeak'] = 5
    df.loc[df['oldpeak'] < 0, 'oldpeak'] = 0
    
    # 8. Balancear clases (Submuestreo de la clase mayoritaria)
    count_no, count_yes = df['heart_disease'].value_counts()
    print(f"Distribución original antes de balancear: Sin enfermedad={count_no}, Con enfermedad={count_yes}")
    
    df_class_no = df[df['heart_disease'] == 0]
    df_class_yes = df[df['heart_disease'] == 1]
    
    no_downsampled = resample(df_class_no,
                              replace=False,
                              n_samples=count_yes,
                              random_state=27)
    df_balanceado = pd.concat([df_class_yes, no_downsampled]).reset_index(drop=True)
    print(f"Distribución balanceada: {df_balanceado['heart_disease'].value_counts().to_dict()}")
    
    # 9. Separar variables explicativas (X) y objetivo (y)
    X = df_balanceado.drop(columns=['heart_disease'])
    y = df_balanceado['heart_disease']
    
    # Asegurar tipos correctos para get_dummies
    for col in ['fbs', 'exang']:
        X[col] = X[col].astype(bool)
        
    # Dummy encoding
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_columns = X_encoded.columns.tolist()
    print(f"Características codificadas: {len(feature_columns)} columnas.")
    
    # Guardar las columnas necesarias para el modelo
    os.makedirs('models', exist_ok=True)
    with open('models/model_columns.json', 'w') as f:
        json.dump(feature_columns, f)
        
    with open('models/pipeline_params.json', 'w') as f:
        json.dump(pipeline_params, f)
        
    # 10. Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 11. Escalado de características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Guardar el escalador
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # 12. Entrenar y guardar los tres modelos
    modelos = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'lr': LogisticRegression(random_state=42, max_iter=1500),
        'dt': DecisionTreeClassifier(max_depth=8, random_state=42)
    }
    
    for name, model in modelos.items():
        model.fit(X_train_scaled, y_train)
        acc = model.score(X_test_scaled, y_test)
        print(f"Modelo {name.upper()} entrenado. Exactitud en Test: {acc:.4f}")
        joblib.dump(model, f'models/{name}_model.joblib')
        
    print("¡Entrenamiento finalizado y modelos guardados en la carpeta models/!")

if __name__ == '__main__':
    main()

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
        raise FileNotFoundError(f"No se encontró: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas.")

    # 2. Variable objetivo binaria
    df['heart_disease'] = (df['num'] > 0).astype(int)

    # 3. Eliminar columnas no necesarias
    df = df.drop(columns=['id', 'num'], errors='ignore')

    # ------------------------------------------------------------------ #
    # 4. Imputación — calcular parámetros ANTES de rellenar              #
    # ------------------------------------------------------------------ #
    numerical_fill = {}
    for col in ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']:
        numerical_fill[col] = float(df[col].median())

    categorical_fill = {}
    for col in ['fbs', 'exang', 'restecg', 'slope', 'thal']:
        if col in ['fbs', 'exang']:
            # Mapear TRUE/FALSE a booleanos
            df[col] = df[col].map(
                {'TRUE': True, 'FALSE': False, True: True, False: False,
                 'True': True, 'False': False}
            )
        val = df[col].mode()[0]
        categorical_fill[col] = val

    # Guardar parámetros de imputación para el API
    pipeline_params = {
        'medians': numerical_fill,
        'modes': {k: (bool(v) if k in ['fbs', 'exang'] else str(v))
                  for k, v in categorical_fill.items()}
    }

    # Rellenar nulos con la sintaxis correcta para pandas ≥ 3 (Copy-on-Write)
    for col, val in numerical_fill.items():
        df[col] = df[col].fillna(val)
    for col, val in categorical_fill.items():
        df[col] = df[col].fillna(val)

    # ------------------------------------------------------------------ #
    # 5. Corrección de valores fisiológicamente imposibles               #
    # ------------------------------------------------------------------ #
    df.loc[df['chol'] == 0, 'chol'] = numerical_fill['chol']
    df.loc[df['trestbps'] == 0, 'trestbps'] = numerical_fill['trestbps']

    # Capping de outliers
    df['trestbps'] = df['trestbps'].clip(upper=200)
    df['chol']     = df['chol'].clip(upper=500)
    df['thalch']   = df['thalch'].clip(lower=60)
    df['oldpeak']  = df['oldpeak'].clip(lower=0, upper=5)

    # Tipos correctos
    df['ca']    = df['ca'].astype(int)
    df['fbs']   = df['fbs'].astype(bool)
    df['exang'] = df['exang'].astype(bool)

    # 6. Eliminar duplicados
    df = df.drop_duplicates().reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 7. Balanceo de clases (downsampling)                               #
    # ------------------------------------------------------------------ #
    count_series = df['heart_disease'].value_counts()
    count_no  = count_series[0]
    count_yes = count_series[1]
    print(f"Distribución original → Sin enfermedad: {count_no}, Con enfermedad: {count_yes}")

    df_class_no  = df[df['heart_disease'] == 0]
    df_class_yes = df[df['heart_disease'] == 1]
    minor_count  = min(count_no, count_yes)

    df_no_down = resample(df_class_no,  replace=False, n_samples=minor_count, random_state=27)
    df_yes_down = resample(df_class_yes, replace=False, n_samples=minor_count, random_state=27)
    df_bal = pd.concat([df_no_down, df_yes_down]).reset_index(drop=True)
    print(f"Distribución balanceada: {df_bal['heart_disease'].value_counts().to_dict()}")

    # ------------------------------------------------------------------ #
    # 8. Codificación y separación X / y                                 #
    # ------------------------------------------------------------------ #
    X = df_bal.drop(columns=['heart_disease', 'age_binned'], errors='ignore')
    y = df_bal['heart_disease']

    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_columns = X_encoded.columns.tolist()
    print(f"Características codificadas: {len(feature_columns)}")

    # Verificar que no queden NaN
    assert not X_encoded.isnull().any().any(), "¡Aún hay NaN en X_encoded!"

    # 9. Guardar metadatos
    os.makedirs('models', exist_ok=True)
    with open('models/model_columns.json', 'w') as f:
        json.dump(feature_columns, f)
    with open('models/pipeline_params.json', 'w') as f:
        json.dump(pipeline_params, f)

    # 10. Split y escalado
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, 'models/scaler.joblib')

    # 11. Entrenamiento
    modelos = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'lr': LogisticRegression(random_state=42, max_iter=1500),
        'dt': DecisionTreeClassifier(max_depth=8, random_state=42),
    }

    for name, model in modelos.items():
        model.fit(X_train_s, y_train)
        acc = model.score(X_test_s, y_test)
        print(f"Modelo {name.upper()} → Accuracy en Test: {acc:.4f}")
        joblib.dump(model, f'models/{name}_model.joblib')

    print("¡Entrenamiento finalizado! Modelos guardados en models/")


if __name__ == '__main__':
    main()

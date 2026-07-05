# ==================== Cell 2 ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import plotly.express as px
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ==================== Cell 4 ====================
# Carga del dataset
# Si usas Google Colab, sube el archivo o monta Drive:
# from google.colab import files; files.upload()
df = pd.read_csv('heart_disease_uci.csv')
df.tail(10)

# ==================== Cell 6 ====================
df.shape

# ==================== Cell 7 ====================
df.info()

# ==================== Cell 9 ====================
df.dtypes

# ==================== Cell 12 ====================
print(df.isnull().sum())

# ==================== Cell 14 ====================
fig = plt.figure(figsize=(8,8))
sns.boxplot(y=df['trestbps'])
plt.title('Distribución de Presión Arterial en Reposo')
plt.show()

# ==================== Cell 16 ====================
# Crear variable binaria para visualización
df['heart_disease'] = (df['num'] > 0).astype(int)
fig = plt.figure(figsize=(8,9))
sns.boxplot(x='heart_disease', y='trestbps', data=df)
plt.title('Presión Arterial según Diagnóstico de Enfermedad Cardíaca')
plt.xlabel('Enfermedad Cardíaca (0=No, 1=Sí)')
plt.show()

# ==================== Cell 18 ====================
df.describe()

# ==================== Cell 20 ====================
df.describe(include=['object'])

# ==================== Cell 22 ====================
df['age'].hist(figsize=(6,6))
plt.xlabel('Edad')
plt.title('Histograma de Edad')
plt.show()

# ==================== Cell 24 ====================
mean = df['age'].mean()
median = df['age'].median()
mode = df['age'].mode()
skew = df['age'].skew()
kurt = df['age'].kurt()

print('La media es:', mean)
print('La mediana es:', median)
print('La moda es:', mode.values)
print('El sesgo es:', skew)
print('La kurtosis es:', kurt)

# ==================== Cell 26 ====================
df['oldpeak'].hist(figsize=(6,6))
plt.xlabel('Oldpeak')
plt.title('Histograma de Oldpeak')
plt.show()

# ==================== Cell 27 ====================
mean = df['oldpeak'].mean()
median = df['oldpeak'].median()
mode = df['oldpeak'].mode()
skew = df['oldpeak'].skew()
kurt = df['oldpeak'].kurt()

print('La media es:', mean)
print('La mediana es:', median)
print('La moda es:', mode.values)
print('El sesgo es:', skew)
print('La kurtosis es:', kurt)

# ==================== Cell 29 ====================
df.duplicated().sum()

# ==================== Cell 31 ====================
df.hist(figsize=(20,15))
plt.suptitle('Histogramas de Variables Numéricas', y=1.02)
plt.tight_layout()
plt.show()

# ==================== Cell 33 ====================
df['age'].hist(figsize=(6,6))
plt.xlabel('Edad')
plt.ylabel('Cantidad')
plt.title('Histograma de Edad', fontweight='bold')
plt.show()

# ==================== Cell 35 ====================
df['thalch'].hist(figsize=(6,6))
plt.xlabel('Frecuencia Cardíaca Máxima')
plt.ylabel('Cantidad')
plt.title('Histograma de Frecuencia Cardíaca Máxima (thalch)', fontweight='bold')
plt.show()

# ==================== Cell 37 ====================
fig = plt.figure(figsize=(7,4), dpi=100)
plt.xticks(rotation=45, fontsize=10)
sns.lineplot(data=df, x='age', y='thalch')
plt.title('Relación entre Edad y Frecuencia Cardíaca Máxima')
plt.xlabel('Edad')
plt.ylabel('Frecuencia Cardíaca Máxima')
plt.tight_layout()
plt.show()

# ==================== Cell 39 ====================
df_cat = df.select_dtypes(include=['object'])
df_cat.head()

# ==================== Cell 40 ====================
for col in df_cat.columns:
    print(f'{col}: \n{df_cat[col].unique()}\n')

# ==================== Cell 42 ====================
# Distribución del diagnóstico principal
sns.countplot(data=df, x='num')
plt.title('Distribución del Diagnóstico de Enfermedad Cardíaca (num)')
plt.xlabel('Nivel de Enfermedad (0=Sano, 1-4=Severidad)')
plt.show()

# ==================== Cell 43 ====================
# Distribución por tipo de dolor en el pecho
sns.countplot(data=df, x='cp')
plt.xticks(rotation=30)
plt.title('Tipos de Dolor en el Pecho (cp)')
plt.show()

# ==================== Cell 44 ====================
# Distribución por origen del dataset
sns.countplot(data=df, x='dataset')
plt.title('Cantidad de Registros por Institución')
plt.show()

# ==================== Cell 46 ====================
male_hd = pd.DataFrame(df.loc[(df['sex'] == 'Male') & (df['heart_disease'] == 1)])
female_hd = pd.DataFrame(df.loc[(df['sex'] == 'Female') & (df['heart_disease'] == 1)])

fig = plt.figure(figsize=(16, 9))

ax = fig.add_subplot(121)
male_pie = pd.DataFrame(male_hd['dataset'].value_counts())
ax.set_title('Institución - Hombres con Enfermedad Cardíaca', fontsize=12)
ax.pie(x=male_pie['count'], labels=male_pie.index, autopct='%.3f%%')

ax = fig.add_subplot(122)
female_pie = pd.DataFrame(female_hd['dataset'].value_counts())
ax.set_title('Institución - Mujeres con Enfermedad Cardíaca', fontsize=12)
ax.pie(x=female_pie['count'], labels=female_pie.index, autopct='%.3f%%')
plt.show()

# ==================== Cell 48 ====================
sns.countplot(data=df, x='sex', hue='heart_disease')
plt.title('Diagnóstico de Enfermedad Cardíaca por Sexo')
plt.show()

# ==================== Cell 49 ====================
#¿Cuáles son las instituciones con menor catidad de pacientes de enfermedad cardíaca?
instituciones_mas_visitas = df[df['num'] == 0]['dataset'].value_counts().reset_index()
instituciones_mas_visitas.columns = ['dataset', 'No of patients']
instituciones_mas_visitas

# ==================== Cell 51 ====================
basemap = folium.Map()
guests_map = px.choropleth(
    instituciones_mas_visitas,
    locations = instituciones_mas_visitas['dataset'],
    color_continuous_scale="portland",
    color = instituciones_mas_visitas['No of patients'],
    hover_name = instituciones_mas_visitas['dataset']
)

guests_map.show()

# ==================== Cell 53 ====================
counts = df['dataset'].value_counts()
plt.subplots(figsize=(10,5))
sns.countplot(x='dataset', hue='sex', data=df)
plt.title('Registros por Institución y Sexo')
plt.show()

# ==================== Cell 55 ====================
plt.figure(figsize=(12,6))
sns.barplot(x='num', y='age', hue='sex', data=df)
plt.title('Edad promedio por Nivel de Diagnóstico y Sexo')
plt.xlabel('Nivel de Diagnóstico (num)')
plt.ylabel('Edad Promedio')
plt.show()

# ==================== Cell 57 ====================
plt.figure(figsize=(15,6))
sns.countplot(data=df, x='age', hue='heart_disease')
plt.xticks(rotation=90, fontsize=7)
plt.title('Distribución de Edad por Diagnóstico')
plt.show()

# ==================== Cell 59 ====================
cp_labels = df['cp'].value_counts().index.tolist()
size = df['cp'].value_counts()
plt.figure(figsize=(8,8))
cmap = plt.get_cmap('Pastel2')
colors = cmap(np.arange(len(cp_labels))*1.0)
my_circle = plt.Circle((0,0), 0.7, color='white')
plt.pie(size, labels=cp_labels, colors=colors, wedgeprops={'linewidth':3,'edgecolor':'white'})
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Tipo de Dolor en el Pecho', weight='bold')
plt.show()

# ==================== Cell 62 ====================
pacientes_cleveland = df[df['dataset'] == 'Cleveland'][df['num'] == 0]['age'].value_counts().reset_index()
pacientes_cleveland.columns = ['Edad', 'N° Pacientes']
pacientes_cleveland

# ==================== Cell 64 ====================
sns.barplot(x = "N° Pacientes", y = "Edad", data = pacientes_cleveland)

# ==================== Cell 66 ====================
pacientes_hungary = df[df['dataset'] == 'Hungary'][df['num'] == 0]['age'].value_counts().reset_index()
pacientes_hungary.columns = ['Edad', 'N° Pacientes']
pacientes_hungary

# ==================== Cell 68 ====================
sns.barplot(x = "N° Pacientes", y = "Edad", data = pacientes_hungary)

# ==================== Cell 70 ====================
counts = df['dataset'].value_counts()
plt.subplots(figsize = (10, 5))
sns.countplot(x= 'dataset', hue='sex', data=df) # Changed 'hue' to 'sex' and removed the incorrect filter
plt.show()

# ==================== Cell 71 ====================
counts = df['dataset'].value_counts()
counts

# ==================== Cell 73 ====================
restecg_cancel = df.groupby('restecg')['heart_disease'].describe()
plt.figure(figsize=(8,6))
sns.barplot(x=restecg_cancel.index, y=restecg_cancel['mean']*100)
plt.title('Influencia del ECG en Reposo en la Enfermedad Cardíaca', fontsize=14)
plt.xlabel('Resultado ECG en Reposo', fontsize=12)
plt.ylabel('% con Enfermedad Cardíaca', fontsize=12)
plt.xticks(rotation=20)
plt.show()

# ==================== Cell 75 ====================
print(df.isnull().sum())

# ==================== Cell 76 ====================
df[['trestbps','chol','fbs','thalch','exang','oldpeak','slope','ca','thal']].describe(include='all')

# ==================== Cell 78 ====================
df = df.drop(['id'], axis=1)

# ==================== Cell 79 ====================
df.info()

# ==================== Cell 81 ====================
# Imputar variables numéricas con mediana
for col in ['trestbps', 'chol', 'thalch', 'oldpeak']:
    df[col].fillna(df[col].median(), inplace=True)

# Imputar variables categóricas con moda
for col in ['fbs', 'exang', 'restecg', 'slope', 'thal']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Para 'ca' usar mediana (variable discreta numérica)
df['ca'].fillna(df['ca'].median(), inplace=True)

print(df.isnull().sum())

# ==================== Cell 83 ====================
df['ca'] = df['ca'].astype('int')
df['fbs'] = df['fbs'].astype('bool')
df['exang'] = df['exang'].astype('bool')
df.dtypes

# ==================== Cell 85 ====================
filter_inconsistent = (df['chol'] == 0) | (df['trestbps'] == 0)
print('Registros inconsistentes:', filter_inconsistent.sum())

# ==================== Cell 86 ====================
# Reemplazar ceros fisiológicamente imposibles con la mediana
df.loc[df['chol'] == 0, 'chol'] = df['chol'].median()
df.loc[df['trestbps'] == 0, 'trestbps'] = df['trestbps'].median()

# Verificación
print('Total registros:', df.shape[0])

# ==================== Cell 87 ====================
# Verificar que todas las variables numéricas tengan valores válidos
filter_valid = df['age'] > 0
df_check = df[filter_valid]
print('Registros válidos:', df_check.shape[0])

# ==================== Cell 89 ====================
columnas = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
n = 1
plt.figure(figsize=(20,15))
for column in columnas:
    plt.subplot(4, 4, n)
    n += 2
    sns.boxplot(y=df[column])
    plt.title(column)
    plt.tight_layout()

# ==================== Cell 91 ====================
df.loc[df['trestbps'] > 200, 'trestbps'] = 200
df.loc[df['chol'] > 500, 'chol'] = 500
df.loc[df['thalch'] < 60, 'thalch'] = 60
df.loc[df['oldpeak'] > 5, 'oldpeak'] = 5
df.loc[df['oldpeak'] < 0, 'oldpeak'] = 0

# ==================== Cell 92 ====================
df['chol'].plot(kind='hist')
plt.title('Distribución de Colesterol tras corrección de outliers')
plt.show()

# ==================== Cell 94 ====================
select_df = df[['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']]

model = LocalOutlierFactor(n_neighbors=30)
y_pred_lof = model.fit_predict(select_df)
print('Total atípicos detectados por LOF:', (y_pred_lof == -1).sum())

# ==================== Cell 95 ====================
outlier_index = (y_pred_lof == -1)
outlier_values = select_df.iloc[outlier_index]
outlier_values.head(10)

# ==================== Cell 97 ====================
plt.figure(figsize=(12,8))
numeric_df = df.select_dtypes(include=['number'])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, linewidths=1, fmt='.2f')
plt.title('Matriz de Correlación')
plt.show()

# ==================== Cell 99 ====================
df.duplicated().sum()

# ==================== Cell 100 ====================
df.loc[df.duplicated(), :]

# ==================== Cell 101 ====================
df_drop = df.drop_duplicates()
df_drop.shape

# ==================== Cell 103 ====================
df_normalize = df_drop.copy()
scaler = MinMaxScaler()
cols_to_scale = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
df_normalize[cols_to_scale] = scaler.fit_transform(df_drop[cols_to_scale])
df_normalize[cols_to_scale].tail(10)

# ==================== Cell 105 ====================
fig = plt.figure(figsize=(8,8))
sns.boxplot(y=df['age'])
plt.title('Distribución de Edad')
plt.show()

# ==================== Cell 106 ====================
grupoEdad = ['Joven', 'Adulto Joven', 'Adulto Medio', 'Adulto Mayor', 'Anciano']
df['age_binned'] = pd.cut(x=df['age'],
                    bins=[0, 35, 45, 55, 65, 100],
                    labels=grupoEdad, include_lowest=True)
df[['age', 'age_binned']].head(10)

# ==================== Cell 107 ====================
sns.catplot(x='age_binned', kind='count', data=df, height=6, aspect=1.5)
plt.title('Distribución por Grupo de Edad')
plt.show()

# ==================== Cell 108 ====================
sns.catplot(x='age_binned', hue='heart_disease', kind='count', data=df, height=6, aspect=1.5)
plt.title('Grupo de Edad vs Diagnóstico de Enfermedad Cardíaca')
plt.show()

# ==================== Cell 110 ====================
df_cat = df_drop.select_dtypes(include=['object', 'bool']).copy()
df_cat['sex'] = df_cat['sex'].map({'Male': 0, 'Female': 1})
df_cat['fbs'] = df_cat['fbs'].astype(int)
df_cat['exang'] = df_cat['exang'].astype(int)
df_cat.head()

# ==================== Cell 112 ====================
df_cat = pd.get_dummies(df_cat, columns=['cp'])
df_cat = pd.get_dummies(df_cat, columns=['restecg'])
df_cat = pd.get_dummies(df_cat, columns=['slope'])
df_cat = pd.get_dummies(df_cat, columns=['thal'])
df_cat = pd.get_dummies(df_cat, columns=['dataset'])
df_cat.head()

# ==================== Cell 114 ====================
sns.countplot(data=df_drop, x='heart_disease')
plt.title('Balance de Clases: Enfermedad Cardíaca')
plt.show()

# ==================== Cell 116 ====================
from sklearn.utils import resample

count_no, count_yes = df_drop['heart_disease'].value_counts()
df_class_no = df_drop[df_drop['heart_disease'] == 0]
df_class_yes = df_drop[df_drop['heart_disease'] == 1]

no_downsampled = resample(df_class_no,
                          replace=False,
                          n_samples=count_yes,
                          random_state=27)
df_balanceado = pd.concat([df_class_yes, no_downsampled])

# ==================== Cell 117 ====================
sns.countplot(data=df_balanceado, x='heart_disease')
plt.title('Distribución Balanceada de Clases')
plt.show()

# ==================== Cell 118 ====================
print(df_balanceado['heart_disease'].value_counts())
print(df_balanceado.shape)

# ==================== Cell 120 ====================
df.to_csv('df.csv', index=False)
print('df.csv guardado!')
df_drop.to_csv('df_drop.csv', index=False)
print('df_drop.csv guardado!')
df_normalize.to_csv('df_normalize.csv', index=False)
print('df_normalize.csv guardado!')
df_balanceado.to_csv('df_balanceado.csv', index=False)
print('df_balanceado.csv guardado!')

# ==================== Cell 122 ====================
df.info()

# ==================== Cell 123 ====================
df_drop.info()

# ==================== Cell 124 ====================
df_normalize.info()

# ==================== Cell 125 ====================
df_balanceado.info()

# ==================== Cell 127 ====================
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Selección de variables predictoras y target
X = df_normalize.drop(columns=['thalch', 'heart_disease', 'num', 'age_binned'], errors='ignore')
y = df_normalize['thalch']

# Codificación de variables categóricas
X = pd.get_dummies(X, drop_first=True)

# División en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputación
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Regresión Lineal
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred_linear = reg_model.predict(X_test)
print('Linear Regression:')
print('MAE:', mean_absolute_error(y_test, y_pred_linear))
print('MSE:', mean_squared_error(y_test, y_pred_linear))
print('R²:', r2_score(y_test, y_pred_linear))
print()

# Ridge
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
print('Ridge Regression:')
print('MAE:', mean_absolute_error(y_test, y_pred_ridge))
print('MSE:', mean_squared_error(y_test, y_pred_ridge))
print('R²:', r2_score(y_test, y_pred_ridge))
print()

# Lasso
lasso_model = Lasso(alpha=0.001)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
print('Lasso Regression:')
print('MAE:', mean_absolute_error(y_test, y_pred_lasso))
print('MSE:', mean_squared_error(y_test, y_pred_lasso))
print('R²:', r2_score(y_test, y_pred_lasso))

# ==================== Cell 128 ====================
# Visualización de métricas para el modelo de regresión
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_ridge, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valor real')
plt.ylabel('Valor predicho')
plt.title('Regresión: Valores predichos vs. reales (Ridge)')
plt.show()

residuals = y_test - y_pred_ridge
plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Regresión: Histograma de residuos')
plt.show()

print('Regresión Ridge:')
print('MSE:', mean_squared_error(y_test, y_pred_ridge))
print('R²:', r2_score(y_test, y_pred_ridge))

# ==================== Cell 130 ====================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              r2_score, classification_report, confusion_matrix)

X = df_balanceado.drop(columns=['heart_disease', 'num', 'age_binned'], errors='ignore')
y = df_balanceado['heart_disease']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf_bin1 = LogisticRegression(random_state=42, max_iter=1500)
clf_bin1.fit(X_train, y_train)
y_pred_logistic = clf_bin1.predict(X_test)

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_logistic))
print('Accuracy:', accuracy_score(y_test, y_pred_logistic))
print('Precision:', precision_score(y_test, y_pred_logistic))
print('Recall:', recall_score(y_test, y_pred_logistic))
print('R2:', r2_score(y_test, y_pred_logistic))
print('Classification Report:')
print(classification_report(y_test, y_pred_logistic))

# ==================== Cell 131 ====================
cm = confusion_matrix(y_test, y_pred_logistic)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión (Regresión Logística)')
plt.show()

# ==================== Cell 133 ====================
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

X = df_balanceado.drop(columns=['heart_disease', 'num', 'age_binned'], errors='ignore')
y = df_balanceado['heart_disease']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf_tree = DecisionTreeClassifier(max_depth=10, min_samples_split=15, random_state=42, criterion='gini')
clf_tree.fit(X_train, y_train)
y_pred_tree = clf_tree.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

cm = confusion_matrix(y_test, y_pred_tree)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión (Decision Tree)')
plt.show()

# ==================== Cell 135 ====================
tree_rules = export_text(clf_tree, feature_names=list(X.columns))
print('Reglas del árbol de decisión:\n', tree_rules[:3000], '...')

plt.figure(figsize=(20,10))
plot_tree(clf_tree, feature_names=list(X.columns), filled=True, rounded=True,
          class_names=['Sin Enfermedad', 'Enfermedad Cardíaca'], max_depth=4)
plt.title('Árbol de Decisión (primeros 4 niveles)')
plt.show()

# ==================== Cell 137 ====================
from sklearn.ensemble import RandomForestClassifier

X = df_balanceado.drop(columns=['heart_disease', 'num', 'age_binned'], errors='ignore')
y = df_balanceado['heart_disease']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf_rf = RandomForestClassifier(n_estimators=120, max_depth=15, min_samples_split=15,
                                 random_state=42, criterion='entropy')
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión (Random Forest)')
plt.show()

# ==================== Cell 139 ====================
# Eliminamos niveles con muy pocos registros para mayor estabilidad
top_levels = df_normalize['num'].value_counts().nlargest(4).index
df_multi = df_normalize[df_normalize['num'].isin(top_levels)]

X = df_multi.drop(columns=['num', 'heart_disease', 'age_binned'], errors='ignore')
y = df_multi['num']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer_multi = SimpleImputer(strategy='mean')
X_train = imputer_multi.fit_transform(X_train)
X_test = imputer_multi.transform(X_test)

clf_multi = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
clf_multi.fit(X_train, y_train)
y_pred_multi = clf_multi.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred_multi))
print(classification_report(y_test, y_pred_multi))

cm = confusion_matrix(y_test, y_pred_multi)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión Multiclase (Nivel de Diagnóstico)')
plt.show()

# ==================== Cell 141 ====================
# =====================================
# MODELO DE REGRESIÓN (Redes Neuronales): Predicción de thalch
# =====================================
#!pip install tensorflow
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              median_absolute_error, explained_variance_score, r2_score)

# Selección de variables predictoras y target
# Usamos df_normalize; excluimos el target 'thalch', variables derivadas y
# variables que no deben usarse como predictoras directas
X = df_normalize.drop(columns=['thalch', 'heart_disease', 'num', 'age_binned'], errors='ignore')
y = df_normalize['thalch']

# Codificación de variables categóricas (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# División en train/test (75% entrenamiento, 25% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Imputación de valores faltantes con la media
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test  = imputer.transform(X_test)

# Escalar los datos (crucial para redes neuronales)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f'Tamaño del conjunto de entrenamiento: {X_train.shape}')
print(f'Tamaño del conjunto de prueba:        {X_test.shape}')

# ==================== Cell 143 ====================
# Definir el modelo de red neuronal
model_reg = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),   # Capa de entrada
    tf.keras.layers.Dense(128, activation='relu'),      # Capa oculta 1
    tf.keras.layers.Dense(64,  activation='relu'),      # Capa oculta 2
    tf.keras.layers.Dense(1)                            # Capa de salida (sin activación → regresión)
])

# Compilar el modelo
# Se usa RMSprop y MSE como pérdida, siguiendo la misma estructura del modelo hotel_bookings
model_reg.compile(optimizer='RMSprop', loss='mse')
# Alternativas comentadas para experimentar:
#model_reg.compile(optimizer='adam', loss='mean_squared_error')
#model_reg.compile(optimizer='sgd',  loss='mean_squared_error')
#model_reg.compile(optimizer='adam', loss=tf.keras.losses.Huber())
#model_reg.compile(optimizer='adam', loss='log_cosh')
#model_reg.compile(optimizer='adam', loss='mean_squared_error',
#                  metrics=['mean_absolute_error', 'root_mean_squared_error'])

# Resumen de la arquitectura
model_reg.summary()

# ==================== Cell 144 ====================
# Entrenar el modelo
history = model_reg.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,   # 10% del train como validación durante el entrenamiento
    verbose=1
)

# Visualizar la curva de pérdida durante el entrenamiento
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'],     label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.title('Curva de Aprendizaje - Red Neuronal (Regresión thalch)')
plt.legend()
plt.tight_layout()
plt.show()

# ==================== Cell 145 ====================
# Realizar predicciones en el conjunto de prueba
y_pred_nn = model_reg.predict(X_test)

# Evaluar el modelo con múltiples métricas
mae              = mean_absolute_error(y_test, y_pred_nn)
mse              = mean_squared_error(y_test, y_pred_nn)
rmse             = mse ** 0.5
medae            = median_absolute_error(y_test, y_pred_nn)
explained_var    = explained_variance_score(y_test, y_pred_nn)
r2               = r2_score(y_test, y_pred_nn)

print('=== Métricas de Evaluación - Red Neuronal (Regresión thalch) ===')
print(f'MAE               : {mae:.4f}')
print(f'MSE               : {mse:.4f}')
print(f'RMSE              : {rmse:.4f}')
print(f'MedAE             : {medae:.4f}')
print(f'Explained Variance: {explained_var:.4f}')
print(f'R²                : {r2:.4f}')

# ==================== Cell 147 ====================
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Diagrama de dispersión: valores predichos vs. valores reales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_nn, alpha=0.5, color='steelblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'k--', lw=2, label='Predicción perfecta')
plt.xlabel('thalch Real (normalizado)')
plt.ylabel('thalch Predicho')
plt.title('Regresión de Redes Neuronales: thalch predicho vs. real')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Histograma de residuos (errores de predicción)
residuals = y_test.values - y_pred_nn.flatten()
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='coral')
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Residuos (Real - Predicho)')
plt.ylabel('Frecuencia')
plt.title('Regresión de Redes Neuronales: Distribución de Residuos')
plt.tight_layout()
plt.show()

# 3. Resumen de métricas finales
print('=== Regresión Redes Neuronales - Métricas Finales ===')
print(f'MSE : {mean_squared_error(y_test, y_pred_nn):.4f}')
print(f'R²  : {r2_score(y_test, y_pred_nn):.4f}')

# ==================== Cell 149 ====================
df.info()

# ==================== Cell 150 ====================
df_drop.info()

# ==================== Cell 151 ====================
df_normalize.info()

# ==================== Cell 152 ====================
df_balanceado.info()

# ==================== Cell 154 ====================
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Seleccionar solo las columnas numéricas relevantes (excluimos id y num/heart_disease)
numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

df_numeric = df_drop[numeric_cols].copy()

# Imputar valores faltantes con la mediana
imputer = SimpleImputer(strategy='median')
df_numeric_imputed = imputer.fit_transform(df_numeric)

# Escalar los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric_imputed)

# --- Aplicar PCA ---
# Se retiene el 85% de la varianza explicada
pca = PCA(n_components=0.85)
df_pca = pca.fit_transform(df_scaled)

print(f"Número de componentes después de PCA: {pca.n_components_}")
print(f"Varianza explicada por los componentes: {np.sum(pca.explained_variance_ratio_):.4f}")

# --- Aplicar MeanShift sobre los datos reducidos por PCA ---
bandwidth_pca = estimate_bandwidth(df_pca, quantile=0.2, n_samples=500)

ms_pca = MeanShift(bandwidth=bandwidth_pca, bin_seeding=True)
ms_pca.fit(df_pca)
clusters_meanshift_pca = ms_pca.labels_
cluster_centers_meanshift_pca = ms_pca.cluster_centers_

n_clusters_meanshift_pca = len(np.unique(clusters_meanshift_pca))
print(f"Número estimado de clústeres después de PCA: {n_clusters_meanshift_pca}")

# Agregar la etiqueta de cluster al DataFrame
df_drop['cluster_meanshift_pca'] = clusters_meanshift_pca

# --- Visualización: PC1 vs PC2 con etiquetas MeanShift ---
df_pca_with_clusters = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(df_pca.shape[1])])
df_pca_with_clusters['cluster_meanshift_pca'] = clusters_meanshift_pca

if df_pca.shape[1] >= 2:
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster_meanshift_pca',
                    data=df_pca_with_clusters, palette='viridis', alpha=0.6)
    plt.title('Clusters de MeanShift sobre PCA (PC1 vs PC2) — Heart Disease')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()

# Resumen de características por cluster
cluster_summary_meanshift_pca = df_drop.groupby('cluster_meanshift_pca')[numeric_cols].mean()
print("\nResumen de características clínicas por cluster (MeanShift sobre PCA):")
print(cluster_summary_meanshift_pca)

print("\nDataFrame con etiquetas de cluster MeanShift sobre PCA:")
print(df_drop.head())

# ==================== Cell 156 ====================
from sklearn.metrics import silhouette_score

# Calcular el Silhouette Score
silhouette_avg = silhouette_score(df_pca, clusters_meanshift_pca)
print(f"El Silhouette Score para los clusters de MeanShift es: {silhouette_avg:.4f}")

# ==================== Cell 158 ====================
print('--- Re-ejecutando PCA y MeanShift con PCA ajustado al 90% de varianza ---\n')

# PCA con 90% de varianza explicada
pca_optimized = PCA(n_components=0.90)
df_pca_optimized = pca_optimized.fit_transform(df_scaled)

print(f"Número de componentes después de PCA (90% varianza): {pca_optimized.n_components_}")
print(f"Varianza explicada: {np.sum(pca_optimized.explained_variance_ratio_):.4f}")

# MeanShift optimizado
bandwidth_pca_optimized = estimate_bandwidth(df_pca_optimized, quantile=0.20, n_samples=500)

ms_pca_optimized = MeanShift(bandwidth=bandwidth_pca_optimized, bin_seeding=True)
ms_pca_optimized.fit(df_pca_optimized)
clusters_meanshift_pca_optimized = ms_pca_optimized.labels_

n_clusters_meanshift_pca_optimized = len(np.unique(clusters_meanshift_pca_optimized))
print(f"Número estimado de clústeres (MeanShift sobre PCA optimizado): {n_clusters_meanshift_pca_optimized}")

# Silhouette Score optimizado
silhouette_avg_optimized = silhouette_score(df_pca_optimized, clusters_meanshift_pca_optimized)
print(f"\nNuevo Silhouette Score (PCA 90% varianza): {silhouette_avg_optimized:.4f}")

# Visualización
if df_pca_optimized.shape[1] >= 2:
    df_pca_optimized_with_clusters = pd.DataFrame(
        df_pca_optimized,
        columns=[f'PC{i+1}' for i in range(df_pca_optimized.shape[1])]
    )
    df_pca_optimized_with_clusters['cluster_meanshift_pca'] = clusters_meanshift_pca_optimized

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster_meanshift_pca',
                    data=df_pca_optimized_with_clusters, palette='viridis', alpha=0.6)
    plt.title('Clusters de MeanShift sobre PCA (PC1 vs PC2) — Varianza 90% — Heart Disease')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()

# ==================== Cell 159 ====================
import pandas as pd
pd.DataFrame(df_pca_optimized_with_clusters).to_csv('df_pca_optimized_with_clusters.csv', index=False)
print('df_pca_optimized_with_clusters.csv guardado correctamente!')

# ==================== Cell 161 ====================
# Valores de quantile a probar
quantile_values = [0.1, 0.2, 0.3, 0.4, 0.5]

results = []

for q in quantile_values:
    print(f"\n--- Probando MeanShift con quantile={q} ---")

    bandwidth_q = estimate_bandwidth(df_pca_optimized, quantile=q, n_samples=500)

    ms_q = MeanShift(bandwidth=bandwidth_q, bin_seeding=True)
    ms_q.fit(df_pca_optimized)
    clusters_meanshift_q = ms_q.labels_

    n_clusters_q = len(np.unique(clusters_meanshift_q))
    print(f"Número estimado de clústeres: {n_clusters_q}")

    if n_clusters_q > 1:
        silhouette_avg_q = silhouette_score(df_pca_optimized, clusters_meanshift_q)
        print(f"Silhouette Score: {silhouette_avg_q:.4f}")
    else:
        silhouette_avg_q = -1.0
        print("Silhouette Score no calculado (menos de 2 clústeres).")

    results.append({
        'quantile': q,
        'n_clusters': n_clusters_q,
        'silhouette_score': silhouette_avg_q
    })

print("\n--- Resumen de Resultados ---")
for r in results:
    print(f"Quantile: {r['quantile']:.1f}, Clusters: {r['n_clusters']}, Silhouette Score: {r['silhouette_score']:.4f}")

best_result = max(results, key=lambda x: x['silhouette_score'])
print(f"\nMejor Silhouette Score con quantile={best_result['quantile']:.1f} ({best_result['silhouette_score']:.4f})")

# ==================== Cell 162 ====================
import pandas as pd
pd.DataFrame(df_pca_optimized_with_clusters).to_csv('df_pca_optimized_with_clusters.csv', index=False)
print('df_pca_optimized_with_clusters.csv guardado correctamente!')

# ==================== Cell 164 ====================
from sklearn.manifold import TSNE

# Aplicar t-SNE a los datos reducidos por PCA (df_pca_optimized)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
df_tsne = tsne.fit_transform(df_pca_optimized)

print("Datos después de aplicar t-SNE sobre PCA:")
print(df_tsne[:5])

# Visualizar resultados de t-SNE
plt.figure(figsize=(10, 8))
if 'cluster_meanshift_pca' in df_drop.columns:
    sns.scatterplot(x=df_tsne[:, 0], y=df_tsne[:, 1],
                    hue=clusters_meanshift_pca_optimized, palette='viridis', alpha=0.6)
    plt.title('Visualización de t-SNE sobre PCA con Clusters de MeanShift — Heart Disease')
    plt.xlabel('Componente t-SNE 1')
    plt.ylabel('Componente t-SNE 2')
else:
    sns.scatterplot(x=df_tsne[:, 0], y=df_tsne[:, 1], alpha=0.6)
    plt.title('Visualización de t-SNE sobre PCA — Heart Disease')
    plt.xlabel('Componente t-SNE 1')
    plt.ylabel('Componente t-SNE 2')

plt.show()

# ==================== Cell 165 ====================
import pandas as pd
pd.DataFrame(df_tsne, columns=['tsne_1', 'tsne_2']).to_csv('df_tsne.csv', index=False)
print('df_tsne.csv guardado correctamente!')

# ==================== Cell 167 ====================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Variable objetivo: heart_disease (binaria: 0 = sin enfermedad, 1 = con enfermedad)
# df_tsne se generó a partir de df_pca_optimized → df_drop, por lo que los índices coinciden
y = df_drop['heart_disease'].values

# X serán los datos transformados mediante t-SNE
X = df_tsne

# Dividir en conjuntos de entrenamiento y prueba (80/20, estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Shape de X_train: {X_train.shape}")
print(f"Shape de X_test:  {X_test.shape}")
print(f"Shape de y_train: {y_train.shape}")
print(f"Shape de y_test:  {y_test.shape}")

# ==================== Cell 168 ====================
# Inicializar y entrenar el modelo de Regresión Logística
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluación del modelo
print("\n--- Evaluación del modelo (t-SNE → Regresión Logística) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred, target_names=['Sin enfermedad', 'Con enfermedad']))

# ==================== Cell 171 ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Preparar datos desde df_balanceado
X_raw = df_balanceado.drop(columns=['heart_disease', 'num', 'age_binned'], errors='ignore')
y_all = df_balanceado['heart_disease']

X_raw = pd.get_dummies(X_raw, drop_first=True)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_raw, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# Pipeline de preprocesamiento (imputación + escalado)
preproc = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
X_train_p = preproc.fit_transform(X_train_r)
X_test_p  = preproc.transform(X_test_r)

print(f"Train: {X_train_p.shape} | Test: {X_test_p.shape}")
print(f"Distribución y_train:\n{y_train_r.value_counts()}")

# ==================== Cell 173 ====================
# Benchmark: DummyClassifier estrategia "most_frequent"
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train_p, y_train_r)
y_pred_dummy = dummy.predict(X_test_p)

print("=== Benchmark (DummyClassifier — Most Frequent) ===")
print(f"Accuracy  : {accuracy_score(y_test_r, y_pred_dummy):.4f}")
print(f"Precision : {precision_score(y_test_r, y_pred_dummy, zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test_r, y_pred_dummy, zero_division=0):.4f}")
print(f"F1-Score  : {f1_score(y_test_r, y_pred_dummy, zero_division=0):.4f}")
print()
print(classification_report(y_test_r, y_pred_dummy,
      target_names=['Sin enfermedad', 'Con enfermedad'], zero_division=0))

# ==================== Cell 175 ====================
modelos = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1500),
    'Decision Tree'      : DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM'                : SVC(probability=True, random_state=42)
}

resultados = {}

for nombre, modelo in modelos.items():
    modelo.fit(X_train_p, y_train_r)
    y_pred  = modelo.predict(X_test_p)
    y_proba = modelo.predict_proba(X_test_p)[:, 1]

    resultados[nombre] = {
        'modelo'   : modelo,
        'y_pred'   : y_pred,
        'y_proba'  : y_proba,
        'accuracy' : accuracy_score(y_test_r, y_pred),
        'precision': precision_score(y_test_r, y_pred),
        'recall'   : recall_score(y_test_r, y_pred),
        'f1'       : f1_score(y_test_r, y_pred),
        'auc'      : roc_auc_score(y_test_r, y_proba)
    }
    print(f"✔ {nombre} entrenado")

# ==================== Cell 177 ====================
# Añadir benchmark a la tabla
y_pred_dummy_proba = np.zeros(len(y_test_r))  # dummy siempre 0 de probabilidad
resultados_tabla = {
    'Benchmark (Dummy)': {
        'accuracy' : accuracy_score(y_test_r, y_pred_dummy),
        'precision': precision_score(y_test_r, y_pred_dummy, zero_division=0),
        'recall'   : recall_score(y_test_r, y_pred_dummy, zero_division=0),
        'f1'       : f1_score(y_test_r, y_pred_dummy, zero_division=0),
        'auc'      : 0.5
    }
}
resultados_tabla.update({k: {m: v[m] for m in ['accuracy','precision','recall','f1','auc']}
                          for k, v in resultados.items()})

df_metricas = pd.DataFrame(resultados_tabla).T
df_metricas.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
df_metricas = df_metricas.sort_values('AUC-ROC', ascending=False)
df_metricas = df_metricas.round(4)

print("=== Comparación de Métricas — Todos los Modelos ===")
print(df_metricas.to_string())

# Visualización tipo heatmap
plt.figure(figsize=(10, 4))
sns.heatmap(df_metricas.astype(float), annot=True, fmt='.4f', cmap='YlGnBu',
            linewidths=0.5, cbar=True)
plt.title('Comparación de Métricas por Modelo', fontsize=14, fontweight='bold')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ==================== Cell 179 ====================
plt.figure(figsize=(9, 7))

# Línea del benchmark (azar)
plt.plot([0, 1], [0, 1], 'k--', label='Benchmark (AUC = 0.50)', linewidth=1.5)

colores = ['royalblue', 'darkorange', 'forestgreen', 'crimson']
for (nombre, res), color in zip(resultados.items(), colores):
    fpr, tpr, _ = roc_curve(y_test_r, res['y_proba'])
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f"{nombre} (AUC = {res['auc']:.4f})")

plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
plt.ylabel('Tasa de Verdaderos Positivos (TPR / Recall)', fontsize=12)
plt.title('Curvas ROC — Comparación de Modelos', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==================== Cell 181 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (nombre, res) in zip(axes, resultados.items()):
    cm = confusion_matrix(y_test_r, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Sin enf.', 'Con enf.'],
                yticklabels=['Sin enf.', 'Con enf.'])
    ax.set_title(f'{nombre}\nAcc={res["accuracy"]:.3f} | AUC={res["auc"]:.3f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor Real')

plt.suptitle('Matrices de Confusión — Todos los Modelos', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

# ==================== Cell 183 ====================
df_bar = df_metricas[['Precision', 'Recall', 'F1-Score', 'AUC-ROC']].copy()

df_bar_melt = df_bar.reset_index().melt(id_vars='index', var_name='Métrica', value_name='Valor')
df_bar_melt.rename(columns={'index': 'Modelo'}, inplace=True)

plt.figure(figsize=(13, 6))
sns.barplot(data=df_bar_melt, x='Modelo', y='Valor', hue='Métrica', palette='Set2')
plt.ylim(0, 1.05)
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Benchmark (0.50)')
plt.title('Comparación de Métricas por Modelo', fontsize=14, fontweight='bold')
plt.xlabel('Modelo')
plt.ylabel('Valor de la Métrica')
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Conclusión automática
mejor = df_metricas['AUC-ROC'].idxmax()
print(f"\n✅ Mejor modelo según AUC-ROC: {mejor}")
print(f"   AUC-ROC  : {df_metricas.loc[mejor, 'AUC-ROC']:.4f}")
print(f"   F1-Score : {df_metricas.loc[mejor, 'F1-Score']:.4f}")
print(f"   Precision: {df_metricas.loc[mejor, 'Precision']:.4f}")
print(f"   Recall   : {df_metricas.loc[mejor, 'Recall']:.4f}")

# ==================== Cell 186 ====================
rf_model = resultados['Random Forest']['modelo']
feature_names = X_raw.columns.tolist()

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Top 15 variables más importantes
top_n = min(15, len(feature_names))
top_idx = indices[:top_n]

plt.figure(figsize=(10, 6))
plt.barh(
    [feature_names[i] for i in top_idx[::-1]],
    importances[top_idx[::-1]],
    color='steelblue'
)
plt.xlabel('Importancia (Gini Impurity Reduction)', fontsize=12)
plt.title(f'Top {top_n} Variables más Importantes — Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Tabla numérica
df_importance = pd.DataFrame({
    'Variable'   : feature_names,
    'Importancia': importances
}).sort_values('Importancia', ascending=False).head(top_n).reset_index(drop=True)

print(df_importance.to_string(index=False))

# ==================== Cell 188 ====================
try:
    import shap
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'shap', '-q'])
    import shap

# Usar el modelo Random Forest (mejor o más interpretable con TreeExplainer)
rf_model = resultados['Random Forest']['modelo']

# TreeExplainer es nativo para modelos de árbol (rápido y exacto)
explainer_shap = shap.TreeExplainer(rf_model)
shap_values    = explainer_shap.shap_values(X_test_p)

# shap_values puede ser lista [clase_0, clase_1]; tomamos clase 1 (Con enfermedad)
# Aseguramos que sv sea siempre (num_samples, num_features)
if isinstance(shap_values, list):
    sv = shap_values[1] # Para modelos con salida de lista (e.g. multi-output)
else:
    # Para modelos con salida array (num_samples, num_features, num_classes)
    # Seleccionamos los valores SHAP para la clase positiva (índice 1)
    sv = shap_values[:, :, 1]

feature_names_arr = np.array(feature_names)

#Beeswarm plot (resumen global)
plt.figure(figsize=(10, 7))
shap.summary_plot(sv, X_test_p, feature_names=feature_names_arr, show=False)
plt.title('SHAP — Resumen Global de Importancia de Variables', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ==================== Cell 189 ====================
# Selecionar SHAP values for the positive class (class 1)
sv_class_1 = sv[:, :, 1]
shap_mean_abs = np.abs(sv_class_1).mean(axis=0)
df_shap = pd.DataFrame({
    'Variable': feature_names_arr,
    '|SHAP| medio': shap_mean_abs
}).sort_values('|SHAP| medio', ascending=False).head(15).reset_index(drop=True)

plt.figure(figsize=(10, 6))
plt.barh(df_shap['Variable'][::-1], df_shap['|SHAP| medio'][::-1], color='darkorange')
plt.xlabel('Valor |SHAP| medio', fontsize=12)
plt.title('Importancia Global de Variables según SHAP (|SHAP| medio)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print(df_shap.to_string(index=False))

# ==================== Cell 190 ====================
#SHAP Waterfall: explicación individual para el paciente numero 0 del test
print("Explicación SHAP para un paciente individual (índice 0 del test)")
shap_exp = shap.Explanation(
    values        = sv[0, :, 1],
    base_values   = float(explainer_shap.expected_value[1]),
    data          = X_test_p[0],
    feature_names = feature_names_arr
)
shap.waterfall_plot(shap_exp, show=True)

# ==================== Cell 192 ====================
try:
    import lime
    import lime.lime_tabular
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lime', '-q'])
    import lime
    import lime.lime_tabular

# Crear el explicador LIME con el conjunto de entrenamiento
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data  = X_train_p,
    feature_names  = feature_names_arr,
    class_names    = ['Sin enfermedad', 'Con enfermedad'],
    mode           = 'classification',
    random_state   = 42
)

# Explicar la predicción para el paciente numero 0 del test (mismo que en SHAP)
idx_paciente = 0
exp_lime = explainer_lime.explain_instance(
    data_row       = X_test_p[idx_paciente],
    predict_fn     = resultados['Random Forest']['modelo'].predict_proba,
    num_features   = 10
)

print(f"Predicción del modelo para paciente numero {idx_paciente}:")
pred_clase  = resultados['Random Forest']['modelo'].predict([X_test_p[idx_paciente]])[0]
pred_proba  = resultados['Random Forest']['modelo'].predict_proba([X_test_p[idx_paciente]])[0]
print(f"  Clase predicha : {'Con enfermedad' if pred_clase == 1 else 'Sin enfermedad'}")
print(f"  Probabilidad   : {pred_proba[pred_clase]:.4f}")
print(f"  Valor real     : {'Con enfermedad' if y_test_r.iloc[idx_paciente] == 1 else 'Sin enfermedad'}")

# Gráfico LIME
fig_lime = exp_lime.as_pyplot_figure()
plt.title(f'LIME - Explicación Local para Paciente numero {idx_paciente}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Lista de contribuciones ordenadas
print("\nContribuciones LIME (variable → impacto):")
for feat, weight in exp_lime.as_list():
    signo = '▲' if weight > 0 else '▼'
    print(f"  {signo} {feat:40s}  peso = {weight:+.4f}")

# ==================== Cell 194 ====================
from sklearn.metrics import f1_score, recall_score, precision_score

#Reconstruir el DataFrame de test con las variables originales
X_test_orig = X_test_r.copy()
X_test_orig_df = X_test_r.reset_index(drop=True)

#Añadir predicciones y valores reales
y_test_arr  = y_test_r.reset_index(drop=True).values
y_pred_rf   = resultados['Random Forest']['modelo'].predict(X_test_p)

df_sesgo = X_test_orig_df.copy()
df_sesgo['y_real'] = y_test_arr
df_sesgo['y_pred'] = y_pred_rf

#Sesgo por sexo
#La columna 'sex_Male' fue generada por get_dummies; 1 = Hombre, 0 = Mujer
if 'sex_Male' in df_sesgo.columns:
    col_sex = 'sex_Male'
    etiquetas_sex = {1: 'Hombre', 0: 'Mujer'}
else:
    col_sex = None

if col_sex:
    print("Análisis de Sesgo por Sexo")
    for val, etiq in etiquetas_sex.items():
        mask = df_sesgo[col_sex] == val
        if mask.sum() == 0:
            continue
        yr, yp = df_sesgo.loc[mask, 'y_real'], df_sesgo.loc[mask, 'y_pred']
        print(f"  {etiq} (n={mask.sum()}): "
              f"Accuracy={accuracy_score(yr, yp):.4f} | "
              f"Recall={recall_score(yr, yp, zero_division=0):.4f} | "
              f"F1={f1_score(yr, yp, zero_division=0):.4f}")

# ==================== Cell 195 ====================
#Sesgo por grupo de edad
#Recuperar columnas age del df original de test
df_balanceado_reset = df_balanceado.reset_index(drop=True)
X_full_reset = df_balanceado_reset.drop(columns=['heart_disease','num','age_binned'], errors='ignore')
X_full_reset = pd.get_dummies(X_full_reset, drop_first=True)

#Asegurar que age existe
if 'age' in X_full_reset.columns:
    _, X_test_check, _, y_test_check = train_test_split(
        X_full_reset, df_balanceado_reset['heart_disease'],
        test_size=0.2, random_state=42, stratify=df_balanceado_reset['heart_disease']
    )
    age_test = X_test_check['age'].values
    bins_age  = [0, 40, 55, 70, 200]
    labels_age = ['≤40', '41–55', '56–70', '>70']
    grupos_edad = pd.cut(age_test, bins=bins_age, labels=labels_age, right=True)

    print("\nAnálisis de Sesgo por Grupo de Edad")
    df_sesgo_edad = pd.DataFrame({
        'grupo_edad': grupos_edad,
        'y_real': y_test_arr,
        'y_pred': y_pred_rf
    })

    resumen_sesgo = []
    for grupo in labels_age:
        mask = df_sesgo_edad['grupo_edad'] == grupo
        if mask.sum() < 5:
            continue
        yr = df_sesgo_edad.loc[mask, 'y_real']
        yp = df_sesgo_edad.loc[mask, 'y_pred']
        resumen_sesgo.append({
            'Grupo Edad': grupo,
            'N'         : int(mask.sum()),
            'Accuracy'  : round(accuracy_score(yr, yp), 4),
            'Recall'    : round(recall_score(yr, yp, zero_division=0), 4),
            'Precision' : round(precision_score(yr, yp, zero_division=0), 4),
            'F1-Score'  : round(f1_score(yr, yp, zero_division=0), 4)
        })
        print(f"  {grupo:6s} (n={mask.sum():3d}): "
              f"Acc={accuracy_score(yr,yp):.4f} | "
              f"Recall={recall_score(yr,yp,zero_division=0):.4f} | "
              f"F1={f1_score(yr,yp,zero_division=0):.4f}")

    #Gráfico comparativo
    df_sesgo_plot = pd.DataFrame(resumen_sesgo)
    df_sesgo_melt = df_sesgo_plot.melt(
        id_vars='Grupo Edad',
        value_vars=['Accuracy', 'Recall', 'Precision', 'F1-Score'],
        var_name='Métrica', value_name='Valor'
    )
    plt.figure(figsize=(11, 5))
    sns.barplot(data=df_sesgo_melt, x='Grupo Edad', y='Valor', hue='Métrica', palette='Set1')
    plt.ylim(0, 1.05)
    plt.title('Rendimiento del Modelo por Grupo de Edad (Análisis de Sesgo)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Grupo de Edad')
    plt.ylabel('Valor de la Métrica')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


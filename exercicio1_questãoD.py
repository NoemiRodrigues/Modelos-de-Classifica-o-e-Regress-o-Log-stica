'''d.Aplique a regressão logística para avaliar o modelo de classificação.'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df_iris = pd.read_csv('iris.csv')

coluna_alvo = next((col for col in ['Species'] if col in df_iris.columns), None)

X = df_iris.select_dtypes(include=np.number)  # seleciona colunas numéricas
y = df_iris[coluna_alvo]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

print("\nMatriz de Confusão:")
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
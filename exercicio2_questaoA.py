'''2. Utilizando o dataset load_digits. Exemplo de como fazer a
importação do dataset usando o sklearn:'''
'''from sklearn.datasets import
load_digits digits = load_digits()
Responda:
a. Faça uma análise inicial sobre esse
dataset:
i. Quantos dados possui?
ii. Existem dados nulos? Se sim quantos?
iii. Todos são dados numéricos ou existem colunas com dados
categóricos?'''


from sklearn.datasets import load_digits
import pandas as pd
import numpy as np

digits = load_digits()

df_digits = pd.DataFrame(digits.data)
df_digits['target'] = digits.target

num_linhas, num_colunas = df_digits.shape
print(f"O dataset possui {num_linhas} amostras e {num_colunas} colunas (features + target).")

num_nulos = df_digits.isnull().sum().sum()
print(f"Quantidade total de dados nulos: {num_nulos}")

tipos_colunas = df_digits.dtypes.unique()
print(f"Tipos de dados presentes no dataset: {tipos_colunas}")

print("\nDescrição rápida das primeiras linhas:")
print(df_digits.head(5))
'''i. Quantos dados possui?'''

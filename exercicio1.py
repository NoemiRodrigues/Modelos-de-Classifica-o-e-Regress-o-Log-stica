import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#a. Faça uma análise inicial sobre esse dataset 
df_iris = pd.read_csv("iris.csv")

print (df_iris.head(5))

print("\nInformações gerais:")
print(df_iris.info())

print("\nValores nulos por coluna:")
print(df_iris.isnull().sum())

print("\nEstatísticas descritivas:")
print(df_iris.describe())


print("\nNomes das colunas:")
print(df_iris.columns)


coluna_alvo = ['Species']
coluna_alvo = next(filter(lambda col: col in df_iris.columns, coluna_alvo), None)
assert coluna_alvo is not None, "Coluna de espécie não encontrada no dataset."
print(f"\nClasses da coluna '{coluna_alvo}':")
print(df_iris[coluna_alvo].unique())

#b. Use o boxplot e o histograma para caracterizar as propriedades de cada uma das espécies existentes
plt.figure(figsize=(12, 8))
for i, coluna in enumerate(df_iris.select_dtypes(include=np.number).columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(
        data=df_iris,
        x=coluna_alvo,
        y=coluna,
        hue=coluna_alvo,
        palette="Set2",
        legend=False
    )
    plt.title(f'Boxplot de {coluna} por {coluna_alvo}')

plt.tight_layout()
plt.show()

'''c. Somente olhando esses gráficos, é possível afirmar que uma ou mais propriedades (Sepal_Length, Sepal_Width,
Petal_Length, Petal_Width) são suficientes para distinguir as espécies?'''

print ("Olhando os gráficos podemos dizer que as variáveis Petal_Length e Petal_Width são suficientes para distinguir a espécie setosa das demais. Para distinguir Versicolor e Virginica, essas duas variáveis ajudam bastante, mas a separação não é perfeita apenas com uma única variável — uma combinação de atributos (como acontece nos classificadores) é mais eficiente.Sepal_Length e Sepal_Width não são suficientes, embora possam contribuir em combinação com as outras.")





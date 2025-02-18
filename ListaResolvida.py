##############################################################################################################
##############################################################################################################
print("#"*100)
print("ATIVIDADE 1 - AQUISIÇÃO DE DADOS E LEITURA")
print("#"*100)
# SOLICITADO 1: Escreva um script em Python que leia um arquivo CSV contendo dados de vendas de uma empresa. O arquivo deve conter as seguintes colunas: Data, Produto, Quantidade, Preço. 
# SOLICITADO 2: Utilize a biblioteca pandas para ler o arquivo 
# SOLICITADO 3: exibir as primeiras 5 linhas do DataFrame resultante.

##############################################################################################################
# SOLICITADO 2: Utilize a biblioteca pandas para ler o arquivo.
#
# RESOLUÇÃO:
# Importando a biblioteca pandas
import pandas as pd

# Definindo o caminho/nome do arquivo
arquivo = "sales_data_sample.csv"

# Lendo o arquivo csv
# Primeiro problema, o arquivo contem um probelema de encoding, então foi necessário identificar o encoding correto e então passar como parametro para o read_csv.
dataframe = pd.read_csv(arquivo, encoding="Windows-1252")

##############################################################################################################
# SOLICITADO 1: O arquivo deve conter as seguintes colunas: "Data, Produto, Quantidade, Preço".
#
# RESOLUÇÃO:
# Vamos então selecionando as colunas relevantes e renomeando
# Segundo problema, o arquivo não contem exatammente todas as colunas solicitaddas, sendo assim, foi necessário selecionar apenas as colunas solicitadas.
# PRODUTO não estava especificado no arquivo, então foi utilizado a coluna "PRODUCTLINE" como sendo a mais próxima.
dataframe = dataframe[["ORDERDATE", "PRODUCTLINE","QUANTITYORDERED", "PRICEEACH"]].rename(
    columns={"ORDERDATE": "Data", "PRODUCTLINE":"Produto", "QUANTITYORDERED": "Quantidade", "PRICEEACH": "Preço"}
)

##############################################################################################################
# SOLICITADO 3: exibir as primeiras 5 linhas do DataFrame resultante.
#
# RESOLUÇÃO:
# Para exibir as primeiras 5 linhas do DataFrame, utilizamos o comando head().

print("-"*50)
print("Primeiras 5 linhas do DataFrame:")
print("-"*50)
print(dataframe.head())
print("-"*50)

##############################################################################################################
##############################################################################################################
print("#"*100)
print("ATIVIDADE 2 - PRÉ PROCESSAMENTO DE DADOS")
print("#"*100)
# SOLICITADO 1: Limpe os dados lidos no exercício anterior removendo linhas com valores nulos
# SOLICITADO 2: Converta a coluna Data para o tipo datetime. 
# SOLICITADO 3: Normalize a coluna Quantidade para que os valores estejam entre 0 e 1.

# Importando a biblioteca pandas
import pandas as pd
# Para normalizar a coluna Quantidade, vamos utilizar o comando MinMaxScaler() da biblioteca sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler

# Importando a biblioteca matplotlib
import matplotlib.pyplot as plt

# Definindo o caminho/nome do arquivo
arquivo = "sales_data_sample.csv"

# Lendo o arquivo csv
dataframe = pd.read_csv(arquivo, encoding="Windows-1252")

# Selecionando as colunas solicitadas e renomeando-as
dataframe = dataframe[["ORDERDATE", "PRODUCTLINE","QUANTITYORDERED", "PRICEEACH"]].rename(
    columns={"ORDERDATE": "Data", "PRODUCTLINE":"Produto", "QUANTITYORDERED": "Quantidade", "PRICEEACH": "Preço"}
)

##############################################################################################################
# SOLICITADO 1: Limpe os dados lidos no exercício anterior removendo linhas com valores nulos
# RESOLUÇÃO:
# Para remover valores nulos, vamos verificar se existem valores nulos no DataFrame utilizando o comando isnull().sum() emm todas as colunas.
print("-"*50)
print("Total de valores nulos por coluna:")
print("-"*50)
print(dataframe.isnull().sum())
print("-"*50)

#Apos a verificação, podemos identificar que não existem valores nulos no DataFrame.
#Mais para fins de aprendizado, vamos executar o comando para remover valores nulos mesmo assim utilizando o comando dropna().
dataframe = dataframe.dropna()

##############################################################################################################
# SOLICITADO 2: Convertendo a coluna Data para o tipo datetime
# RESOLUÇÃO:
# Vamos verificar o tipo de dados da coluna Data antes da conversão.
print("-"*50)
print("Tipo de dados do DataFrame antes da conversão:")
print("-"*50)
print(dataframe.dtypes)
print("-"*50)

# Para converter a coluna Data para o tipo datetime, vamos utilizar o comando pd.to_datetime()
dataframe["Data"] = pd.to_datetime(dataframe["Data"])

# Vamos verificar o tipo de dados da coluna Data após a conversão.
print("-"*50)
print("Tipo de dados do DataFrame após a conversão:")
print("-"*50)
print(dataframe.dtypes)
print("-"*50)

##############################################################################################################
# SOLICITADO 3: Normalize a coluna Quantidade para que os valores estejam entre 0 e 1.
# RESOLUÇÃO:
# Vamos verificar os valores da coluna Quantidade antes da normalização.
print("-"*50)
print("Valores da coluna Quantidade antes da normalização:")
print("-"*50)
print(dataframe["Quantidade"].head())
print("-"*50)

# Para normalizar a coluna Quantidade, vamos utilizar o comando MinMaxScaler() da biblioteca sklearn.preprocessing
# Apos adicionar a biblioteca no inicio do arquivo. 
# Vamos criar um objeto scaler e aplicar a normalização na coluna Quantidade.
scaler = MinMaxScaler()

# A normalização é feita utilizando o comando fit_transform() e passando a coluna Quantidade como parâmetro.
# O resultado da normalização é atribuído a coluna Quantidade.
dataframe["Quantidade"] = scaler.fit_transform(dataframe[["Quantidade"]])

# Vamos verificar os valores da coluna Quantidade após a normalização.
print("-"*50)
print("Valores da coluna Quantidade após a normalização:")
print("-"*50)
print(dataframe["Quantidade"].head())
print("-"*50)

# Exibindo o DataFrame após as alterações
print("-"*50)
print("DataFrame após as alterações:")
print("-"*50)
print(dataframe.head())
print("-"*50)

##############################################################################################################
##############################################################################################################
print("#"*100)
print("ATIVIDADE 3 - ANÁLISE ESTATÍSTICA")
print("#"*100)
# SOLICITADO 1: Calcule a média, mediana, desvio padrão e moda para a coluna Preço do DataFrame resultante do pré-processamento.

##############################################################################################################
# SOLICITADO 1: Calcular a média, mediana, desvio padrão e moda para a coluna Preço.
# RESOLUÇÃO:
# Para calcular a média, vamos utilizar o comando mean() da biblioteca pandas.
media_preco = dataframe["Preço"].mean()
# Para calcular a mediana, vamos utilizar o comando median() da biblioteca pandas.
mediana_preco = dataframe["Preço"].median()
# Para calcular o desvio padrão, vamos utilizar o comando std() da biblioteca pandas.
desvio_padrao_preco = dataframe["Preço"].std()
# Para calcular a moda, vamos utilizar o comando mode() da biblioteca pandas.
moda_preco = dataframe["Preço"].mode()[0]

# Exibindo os resultados
print("-"*50)
print("Estatísticas da coluna Preço:")
print("-"*50)
print(f"Média: {media_preco}")
print(f"Mediana: {mediana_preco}")
print(f"Desvio Padrão: {desvio_padrao_preco}")
print(f"Moda: {moda_preco}")
print("-"*50)

##############################################################################################################
##############################################################################################################
print("#"*100)
print("ATIVIDADE 4 - VISUALIZAÇÃO DE DADOS")
print("#"*100)

# SOLICITADO 1: Crie um gráfico de barras utilizando a biblioteca matplotlib que mostre a quantidade total vendida de cada produto. O eixo x deve representar os produtos e o eixo y a quantidade total vendida.
# RESOLUÇÃO:
# Para criar o gráfico de barras, vamos utilizar a biblioteca matplotlib.

# Calculando a quantidade total vendida de cada produto
quantidade_total_vendida = dataframe.groupby("Produto")["Quantidade"].sum()

# Vamos estabelecer o tamanho da figura, para isso vamos utilizar o comando figure() da biblioteca matplotlib.
plt.figure(figsize=(10, 6))

# Agora vamos definir o gráfico de barras utilizando o comando plot() da biblioteca matplotlib.
# como o solicitado foi um gráfico de barras, vamos utilizar o kind="bar"
quantidade_total_vendida.plot(kind="bar")

# Vamos adicionar o título e rótulos aos eixos
plt.title("Quantidade Total Vendida de Cada Produto")
plt.xlabel("Produto")
plt.ylabel("Quantidade Total Vendida")

# Agora vamos exibir.

print("-"*50)
print("Gráfico de barras mostrando a quantidade total vendida de cada produto:")
print("-"*50)
plt.show()

##############################################################################################################
##############################################################################################################
print("#"*100)
print("ATIVIDADE 5 - K-VIZINHOS MAIS PRÓXIMOS - CLASSIFICAÇÃO")
print("#"*100)

# SOLICITADO 1: Usando a mesma base de dados, crie uma coluna binária chamada Alta_Venda, onde o valor é 1 se a quantidade vendida for maior que a média e 0 caso contrário.
# SOLICITADO 2: Em seguida, crie um modelo de classificação utilizando o algoritmo K-Vizinhos Mais Próximos (KNN) para prever se uma venda será alta ou não.
# SOLICITADO 3: Avalie o modelo utilizando a matriz de confusão.

# Importando a biblioteca pandas
import pandas as pd
# Importando a biblioteca numpy
import numpy as np
# Importando a biblioteca sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Definindo o caminho/nome do arquivo
arquivo = "sales_data_sample.csv"

# Lendo o arquivo csv
dataframe = pd.read_csv(arquivo, encoding="Windows-1252")

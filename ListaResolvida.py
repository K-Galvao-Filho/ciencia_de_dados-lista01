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

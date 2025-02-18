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

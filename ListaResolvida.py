##############################################################################################################
##############################################################################################################
print("#"*100)
print("ATIVIDADE 1 - AQUISIÇÃO DE DADOS E LEITURA")
print("#"*100)
# SOLICITADO 1: Escreva um script em Python que leia um arquivo CSV contendo dados de vendas de uma empresa. 
# O arquivo deve conter as seguintes colunas: Data, Produto, Quantidade, Preço. 
# SOLICITADO 2: Utilize a biblioteca pandas para ler o arquivo 
# SOLICITADO 3: exibir as primeiras 5 linhas do DataFrame resultante.

##############################################################################################################
# SOLICITADO 2: Utilize a biblioteca pandas para ler o arquivo.
#
# RESOLUÇÃO:
# Importando as bibliotecas pandas e outras necessárias
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Definindo o caminho/nome do arquivo
arquivo = "sales_data_sample.csv"

# Lendo o arquivo CSV com os dados de vendas da empresa.
# Tivemos um pequeno problema ao abrir o arquivo, o mesmo estava com uma codificação diferente, sendo assim foi necessário 
# adicionar o encoding="Windows-1252" para que o arquivo fosse aberto corretamente.
dataframe = pd.read_csv(arquivo, encoding="Windows-1252")

##############################################################################################################
# SOLICITADO 1: O arquivo deve conter as seguintes colunas: "Data, Produto, Quantidade, Preço".
#
# RESOLUÇÃO:
# Buscamos selecionar as colunas solicitadas, porém, tivemos novos problemas:
# O arquivo não contem exatammente todas as colunas solicitaddas, sendo assim, foi necessário selecionar apenas as colunas 
# solicitadas. PRODUTO não estava especificado no arquivo, então foi utilizado a coluna "PRODUCTLINE" como sendo a mais próxima.
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

##############################################################################################################
# SOLICITADO 1: Limpe os dados lidos no exercício anterior removendo linhas com valores nulos
#
# RESOLUÇÃO:
# Para remover valores nulos, vamos verificar se existem valores nulos no DataFrame utilizando o comando isnull().sum() 
# emm todas as colunas.
print("-"*50)
print("Total de valores nulos por coluna:")
print("-"*50)
print(dataframe.isnull().sum())
print("-"*50)

#Apos a verificação, podemos identificar que não existem valores nulos no DataFrame.
#Mais para fins de aprendizado, vamos executar mesmo assim o comando para remover valores nulos mesmo assim utilizando 
# o comando dropna().
dataframe = dataframe.dropna()

##############################################################################################################
# SOLICITADO 2: Convertendo a coluna Data para o tipo datetime
#
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
#
# RESOLUÇÃO:
# Vamos verificar os valores da coluna Quantidade antes da normalização.
print("-"*50)
print("Valores da coluna Quantidade antes da normalização:")
print("-"*50)
print(dataframe["Quantidade"])
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
print(dataframe["Quantidade"])
print("-"*50)

# Exibindo o DataFrame após as alterações
print("-"*50)
print("DataFrame após as alterações:")
print("-"*50)
print(dataframe)
print("-"*50)

##############################################################################################################
##############################################################################################################
print("#"*100)
print("ATIVIDADE 3 - ANÁLISE ESTATÍSTICA")
print("#"*100)
# SOLICITADO: Calcule a média, mediana, desvio padrão e moda para a coluna Preço do DataFrame resultante do pré-processamento.

##############################################################################################################
# SOLICITADO: Calcular a média, mediana, desvio padrão e moda para a coluna Preço.
#
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

##############################################################################################################
# SOLICITADO 1: Crie um gráfico de barras utilizando a biblioteca matplotlib que mostre a quantidade total vendida de cada produto. 
# O eixo x deve representar os produtos e o eixo y a quantidade total vendida.
#
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

# Exibindo o gráfico
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

##############################################################################################################
# SOLICITADO 1: Crie uma coluna binária chamada Alta_Venda, onde o valor é 1 se a quantidade vendida for maior que a média e 0 caso contrário.
#
# RESOLUÇÃO:
# Vamos urilizar calcular a média da Quantidade utilizando o comando .mean()
media_quantidade = dataframe["Quantidade"].mean()

# Vamos então criar a coluna Alta_Venda, onde o valor é 1 se a quantidade vendida for maior que a média e 0 caso contrário.
# Para isso, vamos utilizar o comando .astype(int) para converter o resultado da comparação para inteiro.
# Se a quantidade for maior que a média, o valor será 1, caso contrário, será 0.
dataframe["Alta_Venda"] = (dataframe["Quantidade"] > media_quantidade).astype(int)

# Exibindo o DataFrame com a coluna Alta_Venda
print("-"*50)
print("DataFrame com a coluna Alta_Venda:")
print("-"*50)
print(dataframe)
print("-"*50)

##############################################################################################################
# SOLICITADO 2: Crie um modelo de classificação utilizando o algoritmo K-Vizinhos Mais Próximos (KNN) para prever se uma venda será alta ou não.
#
# RESOLUÇÃO:
# Definindo as variáveis independentes (X) e a variável dependente (y)
X = dataframe[["Preço", "Quantidade"]]
y = dataframe["Alta_Venda"]

# Dividindo os dados em conjuntos de treino e teste
# O parâmetro test_size=0.2 indica que 20% dos dados serão utilizados para teste e 80% para treino.
# O parâmetro random_state=42 é utilizado para garantir a reprodutibilidade dos resultados.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo KNN
# foram testados diversos valores de n_neighbors, e foi identificado que o valor 5 foi o que obteve melhor acurácia sem necessidade de ajuste de hiperparametros
# no caso de parametros como 1 e 3, a acurácia foi maior, porem o modelo pode estar sofrendo de overfitting.
# no caso de parametros como 7 e 9, a acurácia foi menor, porem o modelo pode estar sofrendo de underfitting.
# o valor 5 foi o que obteve melhor acurácia sem necessidade de ajuste de hiperparametros
knn = KNeighborsClassifier(n_neighbors=5)

# Treinando o modelo
knn.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
# Vamos utilizar o comando predict() para fazer previsões no conjunto de teste.
y_pred = knn.predict(X_test)


##############################################################################################################
# SOLICITADO 3: Avalie o modelo utilizando a matriz de confusão.
#
# RESOLUÇÃO:
# A matriz de confusão representa o desempenho do modelo de classificação.
# Vamos utilizar a matriz de confusão para avaliar o modelo.
# Para isso, vamos utilizar a biblioteca sklearn.metrics e o comando confusion_matrix().

# Avaliando o modelo utilizando a matriz de confusão e o relatório de classificação
print("-"*50)
print("Matriz de Confusão:")
print("-"*50)
print(confusion_matrix(y_test, y_pred))
print("-"*50)

# O relatório de classificação exibe métricas como precisão, recall, f1-score e suporte.
# Vamos utilizar o comando classification_report() para exibir o relatório de classificação.
print("-"*50)
print("Relatório de Classificação:")
print("-"*50)
print(classification_report(y_test, y_pred))
print("-"*50)

# A acurácia é uma métrica que representa a proporção de previsões corretas do modelo.
# Vamos utilizar o comando accuracy_score() para calcular a acurácia do modelo.
print("-"*50)
print("Acurácia:")
print("-"*50)
print(accuracy_score(y_test, y_pred))
print("-"*50)

##############################################################################################################
# ATIVIDADE 6: Agrupamento - K-means
# SOLICITADO 1: Utilize o algoritmo K-means para agrupar os produtos com base em suas quantidades vendidas e preços.
# SOLICITADO 2: Determine o número ideal de clusters utilizando o método do cotovelo (elbow method).
##############################################################################################################
print("#"*100)
print("ATIVIDADE 6 - AGRUPAMENTO - K-MEANS")
print("#"*100)

# SOLICITADO: Utilize o algoritmo K-means para agrupar os produtos com base em suas quantidades vendidas e preços.
# Determine o número ideal de clusters utilizando o método do cotovelo (elbow method).

# Importando a biblioteca necessária para o K-means

# Definindo os dados para o agrupamento
dados = dataframe[["Quantidade", "Preço"]]

# Determinando o número ideal de clusters utilizando o método do cotovelo
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(dados)
    wcss.append(kmeans.inertia_)

# Plotando o gráfico do método do cotovelo
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Método do Cotovelo (Elbow Method)')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.show()

# A partir do gráfico, escolhemos o número ideal de clusters (por exemplo, 3)
numero_clusters = 3

# Aplicando o K-means com o número ideal de clusters
kmeans = KMeans(n_clusters=numero_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
dataframe['Cluster'] = kmeans.fit_predict(dados)

# Exibindo o DataFrame com os clusters
print("-"*50)
print("DataFrame com os Clusters:")
print("-"*50)
print(dataframe)
print("-"*50)

# Plotando os clusters
plt.figure(figsize=(10, 6))
plt.scatter(dataframe['Quantidade'], dataframe['Preço'], c=dataframe['Cluster'], cmap='viridis')
plt.title('Agrupamento de Produtos')
plt.xlabel('Quantidade')
plt.ylabel('Preço')
plt.show()
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Definindo o caminho/nome do arquivo
arquivo = "DadosVendas.csv"

# Lendo o arquivo CSV com os dados de vendas da empresa.
# Tivemos um pequeno problema ao abrir o arquivo, o mesmo estava com uma codificação diferente, sendo assim foi necessário
# adicionar o encoding="Windows-1252" para que o arquivo fosse aberto corretamente.
dados_vendas = pd.read_csv(arquivo, encoding="Windows-1252")

##############################################################################################################
# SOLICITADO 1: O arquivo deve conter as seguintes colunas: "Data, Produto, Quantidade, Preço".
#
# RESOLUÇÃO:
# Buscamos selecionar as colunas solicitadas, porém, tivemos novos problemas:
# O arquivo não contem exatammente todas as colunas solicitadas, sendo assim, foi necessário selecionar apenas as colunas
# solicitadas. PRODUTO não estava especificado no arquivo, então foi utilizado a coluna "PRODUCTLINE" como sendo a mais próxima.
dados_vendas = dados_vendas[["ORDERDATE", "PRODUCTLINE","QUANTITYORDERED", "PRICEEACH"]].rename(
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
print(dados_vendas.head())
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
# em todas as colunas.
print("-"*50)
print("Total de valores nulos por coluna:")
print("-"*50)
print(dados_vendas.isnull().sum())
print("-"*50)

# Após a verificação, podemos identificar que não existem valores nulos no DataFrame.
# Mas para fins de aprendizado, vamos executar mesmo assim o comando para remover valores nulos utilizando
# o comando dropna().
dados_vendas = dados_vendas.dropna()

##############################################################################################################
# SOLICITADO 2: Convertendo a coluna Data para o tipo datetime
#
# RESOLUÇÃO:
# Vamos verificar o tipo de dados da coluna Data antes da conversão.
print("-"*50)
print("Tipo de dados do DataFrame antes da conversão:")
print("-"*50)
print(dados_vendas.dtypes)
print("-"*50)

# Para converter a coluna Data para o tipo datetime, vamos utilizar o comando pd.to_datetime()
dados_vendas["Data"] = pd.to_datetime(dados_vendas["Data"])

# Vamos verificar o tipo de dados da coluna Data após a conversão.
print("-"*50)
print("Tipo de dados do DataFrame após a conversão:")
print("-"*50)
print(dados_vendas.dtypes)
print("-"*50)

##############################################################################################################
# SOLICITADO 3: Normalize a coluna Quantidade para que os valores estejam entre 0 e 1.
#
# RESOLUÇÃO:
# Vamos verificar os valores da coluna Quantidade antes da normalização.
print("-"*50)
print("Valores da coluna Quantidade antes da normalização:")
print("-"*50)
print(dados_vendas["Quantidade"])
print("-"*50)

# Para normalizar a coluna Quantidade, vamos utilizar o comando MinMaxScaler() da biblioteca sklearn.preprocessing
# Após adicionar a biblioteca no inicio do arquivo.
# Vamos criar um objeto scaler e aplicar a normalização na coluna Quantidade.
escalador = MinMaxScaler()

# A normalização é feita utilizando o comando fit_transform() e passando a coluna Quantidade como parâmetro.
# O resultado da normalização é atribuído à coluna Quantidade.
dados_vendas["Quantidade"] = escalador.fit_transform(dados_vendas[["Quantidade"]])

# Vamos verificar os valores da coluna Quantidade após a normalização.
print("-"*50)
print("Valores da coluna Quantidade após a normalização:")
print("-"*50)
print(dados_vendas["Quantidade"])
print("-"*50)

# Exibindo o DataFrame após as alterações
print("-"*50)
print("DataFrame após as alterações:")
print("-"*50)
print(dados_vendas)
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
media_preco = dados_vendas["Preço"].mean()
# Para calcular a mediana, vamos utilizar o comando median() da biblioteca pandas.
mediana_preco = dados_vendas["Preço"].median()
# Para calcular o desvio padrão, vamos utilizar o comando std() da biblioteca pandas.
desvio_padrao_preco = dados_vendas["Preço"].std()
# Para calcular a moda, vamos utilizar o comando mode() da biblioteca pandas.
moda_preco = dados_vendas["Preço"].mode()[0]

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
quantidade_total_vendida = dados_vendas.groupby("Produto")["Quantidade"].sum()

# Vamos estabelecer o tamanho da figura, para isso vamos utilizar o comando figure() da biblioteca matplotlib.
plt.figure(figsize=(10, 6))

# Agora vamos definir o gráfico de barras utilizando o comando plot() da biblioteca matplotlib.
# Como o solicitado foi um gráfico de barras, vamos utilizar o kind="bar"
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
# Vamos utilizar calcular a média da Quantidade utilizando o comando .mean()
media_quantidade = dados_vendas["Quantidade"].mean()

# Vamos então criar a coluna Alta_Venda, onde o valor é 1 se a quantidade vendida for maior que a média e 0 caso contrário.
# Para isso, vamos utilizar o comando .astype(int) para converter o resultado da comparação para inteiro.
# Se a quantidade for maior que a média, o valor será 1, caso contrário, será 0.
dados_vendas["Alta_Venda"] = (dados_vendas["Quantidade"] > media_quantidade).astype(int)

# Exibindo o DataFrame com a coluna Alta_Venda
print("-"*50)
print("DataFrame com a coluna Alta_Venda:")
print("-"*50)
print(dados_vendas)
print("-"*50)

##############################################################################################################
# SOLICITADO 2: Crie um modelo de classificação utilizando o algoritmo K-Vizinhos Mais Próximos (KNN) para prever se uma venda será alta ou não.
#
# RESOLUÇÃO:
# Definindo as variáveis independentes (X) e a variável dependente (y)
X = dados_vendas[["Preço", "Quantidade"]]
y = dados_vendas["Alta_Venda"]

# Dividindo os dados em conjuntos de treino e teste
# O parâmetro test_size=0.2 indica que 20% dos dados serão utilizados para teste e 80% para treino.
# O parâmetro random_state=42 é utilizado para garantir a reprodutibilidade dos resultados.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo KNN
# Foram testados diversos valores de n_neighbors, e foi identificado que o valor 5 foi o que obteve melhor acurácia sem necessidade de ajuste de hiperparâmetros
# no caso de parâmetros como 1 e 3, a acurácia foi maior, porém o modelo pode estar sofrendo de overfitting.
# no caso de parâmetros como 7 e 9, a acurácia foi menor, porém o modelo pode estar sofrendo de underfitting.
# O valor 5 foi o que obteve melhor acurácia sem necessidade de ajuste de hiperparâmetros
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
##############################################################################################################
print("#"*100)
print("ATIVIDADE 6 - AGRUPAMENTO - K-MEANS")
print("#"*100)

# SOLICITADO 1: Utilize o algoritmo K-means para agrupar os produtos com base em suas quantidades vendidas e preços.
# SOLICITADO 2: Determine o número ideal de clusters utilizando o método do cotovelo (elbow method).
##############################################################################################################

# SOLICITADO 1: Utilize o algoritmo K-means para agrupar os produtos com base em suas quantidades vendidas e preços.
#
# RESOLUÇÃO:
# Definindo os dados para o agrupamento (quantidade e preço).
dados_agrupamento = dados_vendas[["Quantidade", "Preço"]]

# SOLICITADO 2: Determine o número ideal de clusters utilizando o método do cotovelo (elbow method).
#
# RESOLUÇÃO:
# O método do cotovelo é utilizado para determinar o número ideal de clusters.
# Vamos calcular a soma dos quadrados das distâncias (inertia) para diferentes valores de k (número de clusters).
# Em seguida, vamos plotar um gráfico para identificar o ponto de inflexão (cotovelo).
inercia = []
faixa_k = range(1, 11)
for k in faixa_k:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(dados_agrupamento)
    inercia.append(kmeans.inertia_)

# Plotando o gráfico do método do cotovelo
plt.figure(figsize=(10, 6))
plt.plot(faixa_k, inercia, marker='o')
plt.title("Método do Cotovelo (Elbow Method)")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inércia")
plt.show()

# Segundo o gráfico do método do cotovelo, o número ideal de clusters é aquele em que a curva começa a se estabilizar.
# Neste caso, o número ideal de clusters está entre 3 e 4.
# Vamos escolher o número de clusters que melhor se ajusta ao problema. neste caso, vamos escolher 3 clusters.
kmeans = KMeans(n_clusters=3, random_state=42)
dados_vendas["Cluster"] = kmeans.fit_predict(dados_agrupamento)

# Exibindo o DataFrame com a coluna Cluster
print("-"*50)
print("DataFrame com a coluna Cluster:")
print("-"*50)
print(dados_vendas.head())
print("-"*50)

##############################################################################################################
##############################################################################################################
print("#"*100)
print("ATIVIDADE 7 - ANÁLISE DE CLUSTERS")
print("#"*100)

# SOLICITADO 1: Descreva as características de cada cluster encontrado no exercício anterior.
# SOLICITADO 2: Identifique quais produtos estão em cada cluster e discuta possíveis razões para esses agrupamentos.
##############################################################################################################

# SOLICITADO 1: Descreva as características de cada cluster encontrado no exercício anterior.
#
# RESOLUÇÃO:
# Vamos descrever as características de cada cluster utilizando o comando groupby() e o comando describe().
descricao_clusters = dados_vendas.groupby("Cluster")[["Quantidade", "Preço"]].describe()

# Exibindo a descrição dos clusters
print("-"*50)
print("Descrição dos Clusters:")
print("-"*50)
print(descricao_clusters)
print("-"*50)

# SOLICITADO 2: Identifique quais produtos estão em cada cluster e discuta possíveis razões para esses agrupamentos.
#
# RESOLUÇÃO:
# Vamos identificar quais produtos estão em cada cluster utilizando o comando groupby() e o comando apply(list).
produtos_por_cluster = dados_vendas.groupby("Cluster")["Produto"].apply(list)

# Exibindo os produtos em cada cluster


print("-"*50)
print("Com os dados, podemos identificar possíveis possiveis razões para os Agrupamentos:")
print("-"*50)
print("Cluster 0: Esses produtos são luxuosos, mas não tão caros quanto os mais exclusivos. São comprados por colecionadores ou quem tem um hobby, então não são compras frequentes, mas quando acontecem, têm um preço alto.")
print("Cluster 1: Esses produtos, são de luxo e têm preços altos. São comprados esporadicamente por um público restrito, o que justifica o preço elevado e as vendas limitadas.")
print("Cluster 2: Esses produtos, têm preços mais baixos e atraem mais pessoas. A variação de preço é pequena, pois o objetivo é ser acessível. Mesmo com preços baixos, a demanda é constante, pois muitas pessoas procuram opções de transporte mais baratas.")
print("-"*50)


##############################################################################################################
##############################################################################################################
print("#"*100)
print("ATIVIDADE 8 - VISUALIZAÇÃO DE CLUSTERS")
print("#"*100)

# SOLICITADO: Crie uma visualização que mostre os clusters formados pelo algoritmo K-means.
# Utilize um gráfico de dispersão, onde cada ponto representa um produto, e cores diferentes representam os diferentes clusters.
#
# RESOLUÇÃO:
# Vamos criar um gráfico de dispersão utilizando a biblioteca matplotlib.
plt.figure(figsize=(10, 6))
dispersao = plt.scatter(dados_vendas["Quantidade"], dados_vendas["Preço"], c=dados_vendas["Cluster"], cmap="viridis")
plt.title("Clusters de Produtos")
plt.xlabel("Quantidade")
plt.ylabel("Preço")
plt.legend(*dispersao.legend_elements(), title="Clusters")
plt.show()


##############################################################################################################
# SOLICITADO: Realize uma validação cruzada de 5 vezes para o modelo KNN criado no exercício 5.
#
# RESOLUÇÃO:
# A validação cruzada é uma técnica utilizada para avaliar a capacidade de generalização de um modelo.
# Vamos realizar uma validação cruzada de 5 vezes para o modelo KNN.
# Realizando a validação cruzada de 5 vezes
resultados_validacao = cross_val_score(knn, X, y, cv=5)

# Exibindo os resultados da validação cruzada
print("-"*50)
print("Resultados da Validação Cruzada (5 vezes):")
print("-"*50)
print(resultados_validacao)
print("-"*50)
print(f"Média da Acurácia: {resultados_validacao.mean()}")
print(f"Desvio Padrão da Acurácia: {resultados_validacao.std()}")
print("-"*50)
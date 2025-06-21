import pandas
import numpy as np

########################## Cálculo das medianas de cada método para cada índice ##########################

nomes_arquivos = ["rand", "adjusted_rand", "adjusted_mutual_info", "mutual_info", "fowlkes_mallows", "homogeneity_score", "completeness", "v_measure", "silhouette", "calinski", "davies"]

# Abre o arquivo de saída no modo escrita
with open("./Medianas/resultado_medianas.txt", "w") as arquivo_saida:
    arquivo_saida.write("Colunas: kmeans, mst_euclidean, mst_js, hdbscan\n\n")

    for nomes in nomes_arquivos:
        df = pandas.read_csv(f'./Metricas/{nomes}.csv')
        
        dados = df.to_numpy()
        medianas = np.median(dados, axis=0)

        texto_resultado = f"Mediana do {nomes}: {medianas}\n\n"
        arquivo_saida.write(texto_resultado)

print("Resultados salvos em 'resultado_medianas.txt'")

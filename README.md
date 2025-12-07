# Algoritmo Divisivo com Divergência de Jensen–Shannon (MST Clustering)

Implementação de um algoritmo de clustering divisivo baseado em uma MST (árvore geradora mínima), usando Divergência de Jensen–Shannon como métrica de dissimilaridade.  
Projeto realizado em contexto de Iniciação Científica (FAPESP) com objetivo de implementar e analisar o algoritmo.

---

## 🔎 Visão Geral

O objetivo deste projeto é implementar e analisar um método de **agrupamento divisivo baseado em Árvores Geradoras Mínimas (MST)**, utilizando a **Divergência de Jensen–Shannon** como medida de dissimilaridade entre distribuições de dados.

Este trabalho foi desenvolvido no contexto de uma **Iniciação Científica financiada pela FAPESP**, com foco em investigar alternativas aos métodos clássicos de clustering. A abordagem explora estruturas de grafos para identificar padrões globais e separar grupos de maneira robusta.

O método implementado segue três etapas principais:

1. **Conversão das amostras para distribuições de probabilidade**, permitindo o uso adequado da divergência JS.
2. **Cálculo da matriz de dissimilaridade**, aplicando a divergência Jensen–Shannon entre cada par de amostras.
3. **Construção da MST** e **remoção das maiores arestas**, formando clusters ao separar regiões conectadas por ligações fracas.

Essa abordagem é especialmente útil para identificar divisões naturais nos dados e modelar relações estruturais de forma mais sensível do que métodos baseados em centroides.

---

## 🧩 Estrutura do Repositório
CodigosIC/
│
├── Algoritmos/
│ ├── kmeans/
│ │ ├── kMeans.py # Implementação do algoritmo K-Means
│ │ └── test.py # Testes e experimentos usando o K-Means
│ │
│ └── mst_DivisiveClustering/
│ ├── MST_DivisiveClustering.py # Algoritmo completo de clustering divisivo baseado em MST
│ ├── MST_kruskal.py # Implementação do algoritmo de Kruskal para gerar a MST
│ ├── set.py # Estruturas auxiliares (ex: WeightedSet)
│ ├── test.py # Testes do algoritmo MST Divisive Clustering
│ └── unionfind.py # Estrutura Union-Find (Disjoint Set Union)
│
├── Testes/
│ ├── Medianas/ # Experimentos com medianas (códigos adicionais)
│ ├── Metricas/
│ │ ├── hipotese.py # Testes de hipótese sobre as métricas
│ │ ├── main.py # Script principal de execução das métricas
│ │ ├── mediana.py # Cálculo e análise de medianas
│ │ └── script.py # Automação de testes e análises
│
├── Relatorio/
│ └── IC_RelatorioFinal_Matheus.pdf # Relatório final da Iniciação Científica
│
├── .gitignore # Configurações de arquivos ignorados pelo Git
└── README.md # Documentação principal do projeto

---

## 🛠️ Tecnologias & Dependências

- Python  
- NumPy  
- SciPy  
- (Opcional) NetworkX — para construção da MST  
- (Opcional) Matplotlib / seaborn — para visualização  
- Outras bibliotecas descritas em `requirements.txt`

---

## 📊 Resultados

Os experimentos mostram que o algoritmo divisivo baseado em MST apresenta resultados consistentes ao:

* **Identificar clusters bem separados**, mesmo em conjuntos de dados mais complexos;
* **Gerar partições coerentes**, bastando remover um pequeno número de arestas de alto custo;
* **Capturar transições naturais entre grupos**, já que a MST evidencia conexões fracas entre clusters.

Foram realizados testes envolvendo:

* **Métricas estatísticas** (medianas, testes de hipótese);
* **Comparações com outros algoritmos de clustering**, como o K-Means;
* **Análises de estabilidade** sob diferentes configurações e perturbações dos dados.

Os resultados indicam que a combinação **Jensen–Shannon + MST**:

* produz agrupamentos estáveis,
* é sensível à estrutura global dos dados,
* e apresenta desempenho competitivo com técnicas tradicionais, especialmente em cenários onde relações entre pontos são melhor modeladas como grafos.

--

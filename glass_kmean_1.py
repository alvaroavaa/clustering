import matplotlib
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics


glass = pd.read_csv('glassClass.csv')

le = LabelEncoder()
le.fit(glass['Type'])
#print(list(le.classes_)) # identifica os tipos de categorias existentes
glass['Type'] = le.transform(glass['Type']) # transforma as espécies em um valor inteiro. 

g_matrix = glass[['RI','Na','Mg','Al','Si','Ca','Ba','Fe']].values
cluster_model = KMeans(n_clusters=7, random_state=10) # n_clusters = significa que o algoritmo K-means tentará dividir os dados em três clusters distintos.  # random_state = parâmetro que controla a aleatoriedade do algoritmo.
cluster_model = cluster_model.fit(g_matrix)
#print(cluster_model.labels_) # mostra como o algoritmo classificou todos os pontos

cluster_label = cluster_model.fit_predict(g_matrix)
glass['pred'] = cluster_label #apenas para criar uma coluna de preditores (ele salva os valores que foram classificados pelo cluster no Kmeans)

#plot dos dados até aqui
#sns.FacetGrid(glass,hue='pred').map(plt.scatter,'RI','Na').add_legend()
#plt.show()


#calculando a métrica de performance
metrica = sm.accuracy_score(glass.Type, cluster_model.labels_)

metrica_ajust = metrics.adjusted_rand_score(glass.Type, cluster_model.labels_)
print(metrica_ajust) # índice de rand ajustado é uma função que mede a similaridade das duas atribuições

# Se o valor da métrica ficar baixo, neste caso 0.27, o que podemo fazer para melhorar? 
# podemos alterar o n_clusters para que aumente a acurácia dos grupos. 

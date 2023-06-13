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

iris = sns.load_dataset("iris")

#print(iris.head(5))
#print(iris.species.value_counts()) #conta o valor total de cada uma dos grupos a coluna species. 

le = LabelEncoder()
le.fit(iris['species'])
#print(list(le.classes_)) # identifica os tipos de espécies existentes

iris['species'] = le.transform(iris['species']) # transforma as espécies em um valor inteiro. 

#print(iris['species'][100:105])

iris_matrix = iris[['sepal_length','sepal_width','petal_length','petal_width']].values # separa os valores da matriz

cluster_model = KMeans(n_clusters=3, random_state=10) # n_clusters = significa que o algoritmo K-means tentará dividir os dados em três clusters distintos.  # random_state = parâmetro que controla a aleatoriedade do algoritmo.
cluster_model = cluster_model.fit(iris_matrix)

#print(cluster_model.labels_) # separa as classes de dados nos três clusters baseado no ganho. 

cluster_labels = cluster_model.fit_predict(iris_matrix)
#print(cluster_labels)

#criando uma nova coluna para os predictores
iris['pred'] = cluster_labels
print(iris)

sns.FacetGrid(iris,hue='species').map(plt.scatter,'sepal_length','sepal_width').add_legend()
plt.show()


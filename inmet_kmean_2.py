from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import utils
import pandas as pd
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

#Lendo o arquivo csv da estação
data = pd.read_csv('/home/alvaro/programacao/cursos/clustering/dados_A701_H_2022-01-01_2022-12-31.csv', sep=';')

# Remove rows with NaN values
data = data.dropna(subset=['PRECIPITACAO TOTAL, HORARIO(mm)','PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA(mB)','UMIDADE RELATIVA DO AR, HORARIA(%)','TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)']) # se quiser remover apenas de uma variável, usar o v]

#Transpõe as colunas e calcula alguns parâmetros estatísticos
transpose = data.describe().transpose()

#selecione as variáveis para a clusterização
features = ['PRECIPITACAO TOTAL, HORARIO(mm)','PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA(mB)','UMIDADE RELATIVA DO AR, HORARIA(%)','TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)']
newdata = data[features]

X = StandardScaler().fit_transform(newdata)

#criando a interação do Kmeans
kmeans = KMeans(n_clusters=12)
model = kmeans.fit(X)
print("model\n", model)

centers = model.cluster_centers_
print(centers)

# Function that creates a DataFrame with a column for Cluster Number, a coluna é nomeada de prediction
def pd_centers(featuresUsed, centers):
	colNames = list(featuresUsed)
	colNames.append('prediction')

	# Zip with a column called 'prediction' (index)
	Z = [np.append(A, index) for index, A in enumerate(centers)]

	# Convert to pandas data frame for plotting
	P = pd.DataFrame(Z, columns=colNames)
	P['prediction'] = P['prediction'].astype(int)
	return P


# Function that creates Parallel Plots
def parallel_plot(data):
	my_colors = list(islice(['b','r','g','y','k','c','orange','magenta','purple','brown','pink','cyan'], None, len(P)))
	plt.figure(figsize=(15,8)).gca().axes.set_ylim([-30,+30])
	parallel_coordinates(P, 'prediction', color = my_colors, marker='o')


P = pd_centers(features, centers)

parallel_plot(P[P['UMIDADE RELATIVA DO AR, HORARIA(%)'] < -.5])

plt.show()

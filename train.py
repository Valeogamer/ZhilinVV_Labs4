import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv('iris.data', header=None)
data.pop(4)


method_Kmeans = KMeans(n_clusters = 3)
method_Kmeans.fit(data)
y_pred = method_Kmeans.predict(data)


with open('predict.txt', 'w') as f:
    print(*y_pred, sep = '\n', end='', file=f)
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score

y_true = pd.read_csv('iris.data', header = None).pop(4)
y_pred = pd.read_csv('predict.txt', header = None)

dictionary = { 
    'Iris-setosa' : 1,
    'Iris-versicolor' : 0,
    'Iris-virginica' : 2
}

y_true = y_true.map(dictionary)

acc = accuracy_score(y_true, y_pred)
print(acc)
with open ('metrics.json', 'w') as f:
    json.dump(
        {"ACCURACY" : acc},
        f
    )
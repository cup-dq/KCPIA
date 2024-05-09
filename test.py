import pandas as pd
from sklearn.model_selection import  StratifiedKFold
import numpy as np
from KCIA import KCIA
dataset = pd.read_csv('ecoli1.csv', encoding='utf-8', delimiter=",")
dataset = pd.DataFrame(dataset)
dataset = np.array(dataset)
X = dataset[:, :-1]
y = dataset[:, -1]
kf = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X, y):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    lclf = KCIA()
    X_resampled, y_resampled = lclf.fit_resample(X_train, y_train, n_nbor=3, Mutation_rate=0.1,
                                                 Crossover_rate=0.8)
    print(X_resampled)
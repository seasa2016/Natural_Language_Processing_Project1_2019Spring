import numpy as np
from sklearn import metrics

true_y, pred_y = np.array([2,1]),np.array([2,2])

w = [1.0 / 15,1.0 / 5,1.0 / 16]
sample_weight = [w[i] for i in true_y]
weighted_acc = metrics.accuracy_score(true_y, pred_y, normalize=True, sample_weight=sample_weight)

print(weighted_acc)
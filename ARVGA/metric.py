from sklearn import metrics
from munkres import Munkres
import numpy as np


def cluster_accuracy(pred, labels, num):
    l1 = l2 = range(num)
    cost = np.zeros((num, num), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(labels) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(labels, new_predict)
    f1_macro = metrics.f1_score(labels, new_predict, average='macro')
    precision_macro = metrics.precision_score(labels, new_predict, average='macro')
    recall_macro = metrics.recall_score(labels, new_predict, average='macro')
    f1_micro = metrics.f1_score(labels, new_predict, average='micro')
    precision_micro = metrics.precision_score(labels, new_predict, average='micro')
    recall_micro = metrics.recall_score(labels, new_predict, average='micro')
    nmi = metrics.normalized_mutual_info_score(labels, pred, average_method='geometric')
    return acc, nmi, f1_macro

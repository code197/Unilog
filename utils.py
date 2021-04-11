import os
import copy
import csv
import json
import logging
from sklearn.metrics import f1_score,precision_score,recall_score, confusion_matrix
import numpy as np

### preds, labels都是1维array
def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    acc = (preds == labels).mean()
    p = precision_score(y_true=labels, y_pred=preds)
    r = recall_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds)
    m = confusion_matrix(y_true=labels, y_pred=preds)
    fpr = m[0, 1] / (np.sum(m[0, :]))

    return {"acc": acc, "p": p, "r": r, "f1": f1, "fpr": fpr}

def criterion_metrics(result):
    return result["f1"]
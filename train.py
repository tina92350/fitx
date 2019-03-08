#!/usr/bin/env python3
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def get_data(dataset, s_date='2010-01-01', e_date='2016-12-31', features=['high', 'low', 'open', 'close']):
    x, y = [], []
    for date in list(dataset.keys()):
        if s_date <= date <= e_date:
            tmp = []
            for i in features:
                tmp.append(dataset[date][i])
            x.append(tmp)
            y.append(dataset[date]['y'])
    return np.array(x), np.array(y)

def clf(tr_x, tr_y):
    param = {
        'max_depth': 10,
    }
    clf = XGBClassifier(**param)
    clf.fit(tr_x, tr_y)

    return clf

if '__main__' == __name__:
    with open('./tmp/data.pkl', 'rb') as f:
        dataset = pickle.load(f)
    tr_x, tr_y = get_data(dataset)
    te_x, te_y = get_data(dataset, '2017-01-01', '2018-06-15')

    clf = clf(tr_x, tr_y)
    pred = clf.predict(te_x)

    print(classification_report(te_y, pred))

#!/usr/bin/env python3
import numpy as np
import pickle
import itertools

from sklearn.metrics import classification_report
from train import get_data
from xgboost import XGBClassifier

with open('./tmp/data_60_20.pkl', 'rb') as f:
    dataset = pickle.load(f)
with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)

#features = ['bias_10', 'bias_10_label', 'rsv', 'ma_5_20', 'ma_5_20_label', 'ma_10_20_label', 'rsi_label', 'volume', 'mtm_10', 'psy_5', 'k_label', 'rsi_6', 'mtm_6', 'rsi_12', 'vol_6', 'k_value', 'psy_10', 'vol_12_label', 'd_value', 'kd_label', 'vol_12', 'macd_9', 'vol_6_label', 'ma_10', 'ma_5', 'ma_25', 'ma_6']
features = ['bias_10', 'bias_10_label', 'ma_5_20', 'ma_10_20_label', 'rsi_label', 'volume', 'rsv', 'ma_5_20_label', 'mtm_6', 'kd_label', 'psy_5', 'rsi_6', 'vol_12_label', 'k_value', 'psy_10', 'mtm_10', 'vol_6', 'k_label', 'rsi_12', 'vol_6_label', 'macd_9', 'vol_12', 'd_value', 'ma_5', 'ma_10']

#tr_x, tr_y, _ = get_data(dataset, features, '2009-01-01', '2016-12-31')
te_x, te_y, date = get_data(dataset, features, '2017-01-01', '2018-06-05')

#clf = XGBClassifier(max_depth=5, n_estimators=120)
#clf.fit(tr_x, tr_y)
pred = clf.predict(te_x)
print(classification_report(te_y, pred))

with open('./tmp/m_y.pkl', 'rb') as f:
    m_y = pickle.load(f)
    
    
earn = 0
arr = []
for i in range(len(te_x)):
    xx = m_y[date[i]]
    x_min = np.argsort(xx)[0] #min
    x_max = np.argsort(xx)[-1] #max
    
    pred = clf.predict(te_x[i].reshape(1, -1))
    ans = te_y[i]
    
    if pred==1 and ans==1:
        earn += 60-(xx[0]- xx[x_min])
    elif pred==-1 and ans==-1:
        earn += 60-(xx[x_max]-xx[0])

    elif pred*ans == -1:
        earn -= 20
    else:
        earn += m_y[date[i]][-1] - m_y[date[i]][0]
    arr.append(earn)
    
print(earn)
print(arr)
for i in range(len(arr)-1):
    print(arr[i+1] - arr[i])
print(min(arr))



'''
buy_pos = list(set(np.where(pred==1)[0]).intersection(np.where(te_y==1)[0]))
buy_neg = list(set(np.where(pred==1)[0]).intersection(np.where(te_y==-1)[0]))
buy_no = list(set(np.where(pred==1)[0]).intersection(np.where(te_y==0)[0]))

sell_pos = list(set(np.where(pred==-1)[0]).intersection(np.where(te_y==-1)[0]))
sell_neg = list(set(np.where(pred==-1)[0]).intersection(np.where(te_y==1)[0]))
sell_no = list(set(np.where(pred==-1)[0]).intersection(np.where(te_y==0)[0]))

earn = len(buy_pos)*60 + len(sell_pos)*60 - len(buy_neg)*30 - len(sell_neg)*30

for i in date[buy_no]:
    y1 = m_y[i][0]
    y2 = m_y[i][-1]
    earn += y2-y1
    
for i in date[sell_no]:
    y1 = m_y[i][0]
    y2 = m_y[i][-1]
    earn += y2-y1
    
print(earn)

earn1 = 0
for i in date[buy]:
    pos = 0
    neg = 0
    yy = m_y[i]

    for j in range(1, len(yy)):
        if pos<50 and neg<30:
            if yy[j]-yy[j-1] > 0:
                pos += yy[j] - yy[j-1]
            else:
                neg += abs(yy[j] - yy[j-1])
        else:
            break

    if pos >= 50 and neg < 30:
        print('+')
        earn1 += 50
    if pos < 50 and neg >= 30:
        print('-')
        earn1 -= 30
    
earn2 = 0
for i in date[sell]:
    pos = 0
    neg = 0
    yy = m_y[i]
       
    for j in range(1, len(yy)):
        if pos<30 and neg<50:
            if yy[j]-yy[j-1] > 0:
                pos += yy[j] - yy[j-1]
            else:
                neg += abs(yy[j] - yy[j-1])
        else:
            break

    if pos <=30 and neg > 50:
        print('+')
        earn2 += 50
    if pos > 30 and neg <= 50:
        print('-')
        earn2 -= 30


print(earn1)
print(earn2)

'''


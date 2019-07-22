#!/usr/bin/env python3
import numpy as np
import pickle
import itertools
import pandas as pd
from sys import argv
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def get_data(dataset, features, s_date='2009-01-01', e_date='2017-12-31') :
    x, y, d_date = [], [], []
    for date in list(dataset.keys()):
        if s_date <= date <= e_date:
            d_date.append(date)
            tmp = []
            for i in features:
                tmp.append(dataset[date][i])
            x.append(tmp)
            y.append(dataset[date]['y'])
    #print(y.count(1), y.count(0), y.count(-1))
    return np.array(x), np.array(y), np.array(d_date)

def model(tr_x, tr_y):
    param = {
        'max_depth': 4,
        'n_estimators': 180,
    }
    clf = XGBClassifier(**param)
    return clf

def find_features(dataset, features, tr_year, va_year):
    max_precision = 0
    best_features = None
    best_clf = None

    for i in range(1, len(features)):
        print(f'{i} features')
        tr_x, tr_y, _ = get_data(dataset, features[:i], '2009-01-01', f'{tr_year}-12-31')
        va_x, va_y, _ = get_data(dataset, features[:i], f'{va_year}-01-01', f'{va_year}-12-31')

        clf = model(tr_x, tr_y)
        clf.fit(tr_x, tr_y)

        pred = clf.predict(va_x)
        print('acc: ', accuracy_score(va_y, pred))

        report = classification_report(va_y, pred, output_dict=True)
        precision = report['1']['precision'] + report['-1']['precision']
        print('precision: ', precision)

        if precision > max_precision:
            max_precision = precision
            best_features = features[:i]
            best_clf = clf

    return max_precision, best_features, best_clf


if '__main__' == __name__:

    te_year = int(argv[2])
    va_year = te_year - 1
    tr_year = te_year - 2

    with open('./tmp/data_60_%s.pkl'%argv[1], 'rb') as f:
        dataset = pickle.load(f)


    #all_features = ['open', 'close', 'volume', 'vol_6', 'vol_12', 'vol_6_label', 'vol_12_label', 'rsv', 'rsv_5', 'k_value', 'd_value', 'k_label', 'kd_label', 'ma_5', 'ma_6', 'ma_10','ma_25', 'ma_5_20', 'ma_5_20_label', 'ma_10_20_label', 'rsi_6', 'rsi_12', 'rsi_label', 'macd_9', 'bias_10', 'bias_10_label', 'mtm_6', 'mtm_10', 'psy_5', 'psy_10']

    if argv[1] == '30':
        features = ['bias_10','bias_10_label','rsv','ma_5_20','ma_5_20_label','ma_10_20_label','rsi_label','volume','mtm_10','psy_5','k_label','rsi_6','mtm_6','rsi_12','vol_6','k_value','psy_10','vol_12_label','d_value','kd_label','vol_12','macd_9','vol_6_label','ma_10','ma_5','ma_25','ma_6','rsv_5','close','open'] ## data_60_30
    else:
        features = ['bias_10','bias_10_label','ma_5_20','ma_10_20_label','rsi_label','volume','rsv','ma_5_20_label','mtm_6','kd_label','psy_5','rsi_6','vol_12_label','k_value','psy_10','mtm_10','vol_6','k_label','rsi_12','vol_6_label','macd_9','vol_12','d_value','ma_5','ma_10','ma_6','open','ma_25','close','rsv_5'] ## data_60_20

    max_precision, best_features, best_clf = find_features(dataset, features, tr_year, va_year)

    va_x, va_y, _ = get_data(dataset, best_features, f'{va_year}-01-01', f'{va_year}-12-31')
    te_x, te_y, date = get_data(dataset, best_features, f'{te_year}-01-01', f'{te_year}-12-31')

    va_pred = best_clf.predict(va_x)
    te_pred = best_clf.predict(te_x)

    print('validation report: ')
    print(classification_report(va_y, va_pred))

    print('test report: ')
    print(classification_report(te_y, te_pred))

    acc = accuracy_score(te_y, te_pred)
    print('test acc:', acc)

    with open('./m_data.pkl', 'rb') as f:
        m_data = pickle.load(f)

    earn = 0
    arr = []
    for i in range(len(te_x)):
        xx = m_data[date[i]]['Close']
        x_min = np.argsort(xx)[0] #min
        x_max = np.argsort(xx)[-1] #max

        pred = best_clf.predict(te_x[i].reshape(1, -1))
        ans = te_y[i]
        
        if pred == 0:
            continue

        if pred==1 and ans==1:
            earn += 60
        elif pred==-1 and ans==-1:
            earn += 60

        elif pred*ans == -1:
            earn -= int(argv[1])
        else:
            earn += m_data[date[i]]['Close'][-1] - m_data[date[i]]['Open'][0]
        arr.append(earn)

    print(earn)
    print(arr)
    print(min(arr))
    
    print((np.array(date).reshape(-1,1)).shape)
    print((np.array(arr).reshape(-1,1)).shape)
    output = np.concatenate((np.array(date).reshape(-1, 1), np.array(arr).reshape(-1, 1)), axis=1)
    df = pd.DataFrame(output)
    print(output.shape)

    writer = pd.ExcelWriter(f'60_{argv[1]}_{te_year}.xlsx')
    df.to_excel(writer, sheet_name='plot', index=False, header=False)
    writer.save()
    '''
    for i in range(len(te_x)):
        xx = m_y[date[i]]

        pred = best_clf.predict(te_x[i].reshape(1, -1))
        ans = te_y[i]
        if pred==1 and ans==1:
            earn += 60 - m_y[date[i]][0]
        elif pred==-1 and ans==-1:
            earn += 60 - m_y[date[i]][0]

        elif pred*ans == -1:
            earn -= int(argv[1])
        else:
            earn += m_y[date[i]][-1] - m_y[date[i]][0]
        arr.append(earn)

    print(earn)
    #for i in range(len(arr)):
     #   print(arr[i]-arr[i-1])
    print(arr)
    print(min(arr))
    print('arr shape', np.array(arr).shape)
    print('date shape', np.array(date).shape)

    output = np.concatenate((np.array(date).reshape(-1, 1), np.array(arr).reshape(-1, 1)), axis=1)
    df = pd.DataFrame(output)
    print(output.shape)

    writer = pd.ExcelWriter(f'60_{argv[1]}_{te_year}.xlsx')
    df.to_excel(writer, sheet_name='plot', index=False, header=False)
    writer.save()
    '''

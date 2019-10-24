import numpy as np
import pandas as pd
import pickle
import datetime
import itertools
import os
from sys import argv
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid
from database import stockDB
from matplotlib import pyplot

def database_setting():
    config = {
        'host': os.environ.get('stockdb_host'),
        'port': 3306,
        'user': os.environ.get('stockdb_user'),
        'password': os.environ.get('stockdb_passwd'),
        'db': 'fitx',
    }
    return config


def load_database(mydb, start_date, end_date):
    return mydb.exe_query("select * from trading_result where TS_ID = 1 and Date <= \"2019/10/20\"")

def data_split(data):
    train_start = datetime.date(1999, 1, 1)
    train_end = datetime.date(2015, 12, 31)
    validate_start = datetime.date(2016, 1, 1)
    validate_end = datetime.date(2018, 12,31)
    test_start = datetime.date(2018, 1, 1)
    test_end = datetime.date(2018, 12, 31)
    train_date = (data.index > train_start) & (data.index < train_end)
    validate_date = (data.index > validate_start) & (data.index < validate_end)
    test_date = (data.index > test_start) & (data.index < test_end)
    train = data.loc[train_date]
    validate = data.loc[validate_date]
    test = data.loc[test_date]
    tr_x = train.drop(['Date', 'y_today', 'y_to_predict'], axis=1)
    tr_y = train['y_to_predict']
    va_x = validate.drop(['Date', 'y_today', 'y_to_predict'], axis=1)
    va_y = validate['y_to_predict']
    te_x = test.drop(['Date', 'y_today', 'y_to_predict'], axis=1)
    te_y = test['y_to_predict']
    return tr_x, tr_y, va_x, va_y, te_x, te_y, test_start, test_end

def model(tr_x, tr_y, va_x, va_y):
    a = [150, 160, 170, 180]
    b = [3, 4, 5]
    c = [0.1, 0.01]
    max_f1 = 0
    best_clf = None
    for i in a:
        for j in b:
            for k in c:
                model = XGBClassifier(n_estimatores=i, max_depth=j, learning_rate=k)
                model.fit(tr_x, tr_y)
                pred_va = model.predict(va_x)
                report = classification_report(va_y, pred_va, output_dict=True)
                #print(model.feature_importances_)
                f1 = report['1.0']['f1-score'] + report['-1.0']['f1-score']
                #f1 = report['1.0']['f1-score']
                if f1 > max_f1:
                    max_f1 = f1
                    best_clf = model
                    print(i, j, k)
    return best_clf
    

if '__main__' == __name__:

    with open('./df.pkl', 'rb') as f:
        data = pickle.load(f)
    data = data.set_index(data['Date'])
    data = data.fillna(0)
    features = ['Date', 'y_today', 'y_to_predict', 'open-ma5','psy_3','Rise_or_not','ma5-ma10','rsi_6','mtm_6','dji','Close-Low','sp500_point','Volume','vol_avg_6','rsi_12','ma5-ma20_label','Open-Low','ma5-ma20_cross','rsi6-rsi12_cross','High-Low','ma5-ma20','russell','psy_10','moex_point','Close-Open','High-Close','nasdaq','dji_rate','vol_avg_3','ma10-ma20','dif-macd','vol_dif_12','dji_point','ma10-ma20_cross','russell_rate','Open-ma60','sp500','moex_rate','macd_index','ma10-ma20_label','nasdaq_point','vol_dif_6','moex','rsi6-rsi12','High-Open','cac40_point','russell_point','ma5-ma10_label','vol_dif_3','sp500_rate','psy_20','macd','Close-ma5'
]
    for i in data.columns:
        if i not in features:
            data = data.drop(i, axis=1)

    tr_x, tr_y, va_x, va_y, te_x, te_y, test_start, test_end= data_split(data)
    best_clf = model(tr_x, tr_y, va_x, va_y)
    pred_te = best_clf.predict(te_x)
    report = classification_report(te_y, pred_te)
    print(report)

    config = database_setting()
    mydb = stockDB(**config)
    ans = load_database(mydb, test_start, test_end)

    call = pd.Series(ans.Call_result.values, index=[ans.Date.values[i] for i in range(len(ans))]).to_dict()
    put = pd.Series(ans.Put_result.values, index=[ans.Date.values[i] for i in range(len(ans))]).to_dict()
    date = np.array(te_x.index)

    result = {}

    for i in range(len(date)):
        result[date[i]] = pred_te[i]
    
    #print(result)

    #print(call.keys())
    arr = []
    earn = 0
    for i in result.keys():
        if result[i] == 1:
            earn += call[i]
        if result[i] == -1:
            earn += put[i]
        arr.append(earn)
    print(arr)
    #print(earn)

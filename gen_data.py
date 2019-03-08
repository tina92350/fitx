#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pickle
import datetime


def load():
    row = np.loadtxt('tmp/fitx.csv', delimiter=',', skiprows=1, dtype=str)
    data = {
        'Date': np.array([str(datetime.datetime.strptime(i, '%Y/%m/%d').date()) for i in row[:, 0]]),
        'Open': np.array(row[:,1], dtype=float),
        'High': np.array(row[:,2], dtype=float),
        'Low': np.array(row[:,3], dtype=float),
        'Close': np.array(row[:,4], dtype=float),
        'Volume': np.array(row[:,5], dtype=float),
    }

    m_row = np.loadtxt('tmp/fitx.minute.csv', delimiter=',', usecols=(0, 5), skiprows=1, dtype=str)
    m_data = {
        'Date': np.array([str(datetime.datetime.strptime(i, '%Y/%m/%d').date()) for i in m_row[:, 0]]),
        'Close': np.array(m_row[:, 1], dtype=float),
    }

    return data, m_data

def basic(data, dataset):
    for i in range(len(data['Date'])-1):
        dataset[data['Date'][i+1]]['high'] = data['High'][i]
        dataset[data['Date'][i+1]]['low'] = data['Low'][i]
        dataset[data['Date'][i+1]]['open'] = data['Open'][i]
        dataset[data['Date'][i+1]]['close'] = data['Close'][i]
        dataset[data['Date'][i+1]]['volume'] = data['Volume'][i]
    return dataset

def rsv(data, dataset):
    for i in range(1, 9):
        close = data['Close'][i]
        open_1 = data['Open'][i]
        min_9 = np.min(data['Low'][:i])
        max_9 = np.max(data['High'][:i])
        dataset[data['Date'][i+1]]['rsv'] = 100*(close-min_9) / (max_9-min_9)
    for i in range(9, len(dataset)-1):
        close = data['Close'][i]
        open_1 = data['Open'][i]
        min_9 = np.min(data['Low'][i-9:i])
        max_9 = np.max(data['High'][i-9:i])
        dataset[data['Date'][i+1]]['rsv'] = 100*(close-min_9) / (max_9-min_9)
    return dataset


def rsv_5(data, dataset):
    for i in range(1, 5):
        close = data['Close'][i]
        open_1 = data['Open'][i]
        min_5 = np.min(data['Low'][:i])
        max_5 = np.max(data['High'][:i])
        dataset[data['Date'][i+1]]['rsv_5'] = 100*(close-min_5) / (max_5-min_5)
    for i in range(5, len(dataset)-1):
        close = data['Close'][i]
        open_1 = data['Open'][i]
        min_5 = np.min(data['Low'][:i])
        max_5 = np.max(data['High'][:i])
        dataset[data['Date'][i+1]]['rsv_5'] = 100*(close-min_5) / (max_5-min_5)
    return dataset

def kd(data, dataset):
    dataset[data['Date'][0]]['k_value'] = 17.76
    dataset[data['Date'][0]]['d_value'] = 25.08
    dataset[data['Date'][1]]['k_value'] = 17.76
    dataset[data['Date'][1]]['d_value'] = 25.08
    dataset[data['Date'][2]]['k_value'] = 17.76
    dataset[data['Date'][2]]['d_value'] = 25.08
    for i in range(3, len(dataset)-1):
        dataset[data['Date'][i]]['k_value'] = (dataset[data['Date'][i-2]]['k_value'])*2/3 + (dataset[data['Date'][i-1]]['rsv'])*1/3
        dataset[data['Date'][i]]['d_value'] = (dataset[data['Date'][i-2]]['d_value'])*2/3 + (dataset[data['Date'][i-1]]['d_value'])*1/3
    return dataset


#kd交叉處
def kd_label(data, dataset):
    for i in range(2, len(dataset)-2):
        if (dataset[data['Date'][i-1]]['k_value']-dataset[data['Date'][i-1]]['d_value']<0) and (dataset[data['Date'][i]]['k_value']-dataset[data['Date'][i]]['d_value']>=0):
            dataset[data['Date'][i+1]]['kd_label'] = 1
        elif (dataset[data['Date'][i-1]]['k_value']-dataset[data['Date'][i-1]]['d_value']>=0) and (dataset[data['Date'][i]]['k_value']-dataset[data['Date'][i]]['d_value']<0):
            dataset[data['Date'][i+1]]['kd_label'] = -1
        else:
            dataset[data['Date'][i+1]]['kd_label'] = 0
    return dataset

###如果連續3天的d都低於20，要跌了，趕快賣
def d_3_label(dataset):
    for i in range(1, 3):
        dataset[i]['d_3_label'] = 0
    for i in range(3, len(dataset)):
        d_20 = np.array([j['d_value'] for j in dataset])[i-3:i]
        d_20[d_20<=20] = 1
        d_20[d_20>20]=0
    print(d_20)

###如果連續4天的k都超過80，要漲了，趕快買
'''
def kd_value_label(dataset):
    for i in range(1, 3):
        if dataset[:i]['k_value']>=80:
            dataset[i]['kd_value_label'] = 1
        elif dataset[:i]['k_value']<80 and dataset[:i]['d_value']<=20:
            dataset[i]['kd_value_label'] = -1
        else:
            dataset[i]['kd_value_label'] = 0
    for i in range(3, len(dataset)):
        if dataset[i-3:i]['k_value']>=80 and dataset[i-3]['d_value']>20
            dataset[i]['kd_value_label'] = 1
        elif dataset[i-3]['d_value']<=20:
            dataset[i]['kd_value_label'] = -1
        else:
            dataset[i]['kd_value_label'] = 0
'''


def ma_5(data, dataset):
    for i in range(1, 5):
        close = data['Close'][i]
        dataset[data['Date'][i+1]]['ma_5'] = np.mean(data['Close'][:i])
    for i in range(5, len(dataset)-1):
        close = data['Close'][i]
        dataset[data['Date'][i+1]]['ma_5'] = np.mean(data['Close'][i-5:i])
    dataset[data['Date'][1]]['ma_5'] = dataset[data['Date'][2]]['ma_5']
    return dataset

def ma_6(data, dataset):
    for i in range(1, 6):
        close = data['Close'][i]
        dataset[data['Date'][i+1]]['ma_6'] = np.mean(data['Close'][:i])
    for i in range(6, len(dataset)-1):
        close = data['Close'][i]
        dataset[data['Date'][i+1]]['ma_6'] = np.mean(data['Close'][i-6:i])
    return dataset

def ma_10(data, dataset):
    for i in range(1, 10):
        close = data['Close'][i]
        dataset[data['Date'][i+1]]['ma_10'] = np.mean(data['Close'][:i])
    for i in range(10, len(dataset)-1):
        close = data['Close'][i]
        dataset[data['Date'][i+1]]['ma_10'] = np.mean(data['Close'][i-10:i])
    dataset[data['Date'][1]]['ma_10'] = dataset[data['Date'][2]]['ma_10']
    return dataset


def ma_12(data, dataset):
    for i in range(1, 12):
        close = data['Close'][i]
        dataset[data['Date'][i+1]]['ma_12'] = np.mean(data['Close'][:i])
    for i in range(12, len(dataset)-1):
        close = data['Close'][i]
        dataset[data['Date'][i+1]]['ma_12'] = np.mean(data['Close'][i-12:i])
    return dataset

def ma_20(data, dataset):
    for i in range(1, 20):
        close = data['Close'][i]
        dataset[data['Date'][i+1]]['ma_20'] = np.mean(data['Close'][:i])
    for i in range(20, len(dataset)-1):
        close = data['Close'][i]
        dataset[data['Date'][i+1]]['ma_20'] = np.mean(data['Close'][i-20:i])
    dataset[data['Date'][1]]['ma_20'] = dataset[data['Date'][2]]['ma_20']
    return dataset

def ma_25(data, dataset):
    for i in range(1, 25):
        close = data['Close'][i]
        dataset[data['Date'][i+1]]['ma_25'] = np.mean(data['Close'][:i])
    for i in range(25, len(dataset)-1):
        close = data['Close'][i]
        dataset[data['Date'][i+1]]['ma_25'] = np.mean(data['Close'][i-25:i])
    return dataset

def ma_60(data, dataset):
    for i in range(1, 60):
        close = data['Close'][i]
        dataset[data['Date'][i+1]]['ma_60'] = np.mean(data['Close'][:i])
    for i in range(60, len(dataset)-1):
        close = data['Close'][i]
        dataset[data['Date'][i+1]]['ma_60'] = np.mean(data['Close'][i-60:i])
    return dataset

# 5MA 穿過 20MA
def ma_5_20_label(data, dataset):
    dataset[data['Date'][1]]['ma_5_20_label'] = 0
    dataset[data['Date'][2]]['ma_5_20_label'] = 0
    for i in range(3, len(dataset)-1):
        if (dataset[data['Date'][i-2]]['ma_5']-dataset[data['Date'][i-2]]['ma_20']<0) and (dataset[data['Date'][i-1]]['ma_5']-dataset[data['Date'][i-1]]['ma_20']>=0):
            dataset[data['Date'][i]]['ma_5_20_label'] = 1
        elif (dataset[data['Date'][i-2]]['ma_5']-dataset[data['Date'][i-2]]['ma_20']>0) and (dataset[data['Date'][i-1]]['ma_5']-dataset[data['Date'][i-1]]['ma_20']<=0):
            dataset[data['Date'][i]]['ma_5_20_label'] = -1
        else:
            dataset[data['Date'][i]]['ma_5_20_label'] = 0
    return dataset

# 10MA 穿過 20MA
def ma_10_20_label(data, dataset):
    dataset[data['Date'][1]]['ma_10_20_label'] = 0
    dataset[data['Date'][2]]['ma_10_20_label'] = 0
    for i in range(3, len(dataset)-1):
        if (dataset[data['Date'][i-2]]['ma_10']-dataset[data['Date'][i-2]]['ma_20']<0) and (dataset[data['Date'][i-1]]['ma_10']-dataset[data['Date'][i-1]]['ma_20']>=0):
            dataset[data['Date'][i]]['ma_10_20_label'] = 1
        elif (dataset[data['Date'][i-2]]['ma_10']-dataset[data['Date'][i-2]]['ma_20']>0) and (dataset[data['Date'][i-1]]['ma_10']-dataset[data['Date'][i-1]]['ma_20']<=0):
            dataset[data['Date'][i]]['ma_10_20_label'] = -1
        else:
            dataset[data['Date'][i]]['ma_10_20_label'] = 0
    return dataset

# 5MA and 10MA 同時突破 20MA
def ma_5_10_20_label(dataset):
    dataset[0]['ma_5_10_20_label'] = 0
    dataset[1]['ma_5_10_20_label'] = 0
    for i in range(2, len(dataset)):
        if (dataset[i]['ma_5_20_label'] == 1 and dataset[i]['ma_10_20_label'] == 1):
            dataset[i]['ma_5_10_20_label'] = 1
        elif (dataset[i]['ma_5_20_label'] == -1 and dataset[i]['ma_10_20_label'] == -1):
            dataset[i]['ma_5_10_20_label'] = -1
        else:
            dataset[i]['ma_5_10_20_label'] = 0

def rsi_6(data, dataset):
    updowns = []
    for i in range(1, 6):
        updowns = np.array(data['Close'][:i] - data['Open'][:i])
        rs = (np.sum(updowns[updowns>=0]))/ np.abs(np.sum(updowns[updowns<0])+0.00001)
        dataset[data['Date'][i]]['rsi_6'] = 100*rs/(1+rs)
    for i in range(6, len(dataset)-1):
        updowns = np.array(data['Close'][i-6:i] - data['Open'][i-6:i])
        rs = (np.sum(updowns[updowns>=0]))/ np.abs(np.sum(updowns[updowns<0])+0.00001)
        dataset[data['Date'][i]]['rsi_6'] = 100*rs/(1+rs)
    return dataset
'''
def rsi_6(dataset):
    updowns = []
    for i in range(1, 6):
        updowns = np.array([(j['Open']-j['Close']) for j in dataset[:i]])
        rs = (np.sum(updowns[updowns>=0]))/ np.abs(np.sum(updowns[updowns<0])+0.00001)
        dataset[i]['rsi_6'] = 100*rs/(1+rs)
    for i in range(6, len(dataset)):
        updowns = np.array([(j['Open']-j['Close']) for j in dataset])[i-6:i]
        rs = (np.sum(updowns[updowns>=0]))/ np.abs(np.sum(updowns[updowns<0])+0.00001)
        dataset[i]['rsi_6'] = 100*rs/(1+rs)
'''
def rsi_12(dataset):
    updowns = []
    for i in range(1, 12):
        updowns = np.array([(j['Open']-j['Close']) for j in dataset[:i]])
        rs = (np.sum(updowns[updowns>=0]))/ np.abs(np.sum(updowns[updowns<0])+0.00001)
        dataset[i]['rsi_12'] = 100*rs/(1+rs)
    for i in range(12, len(dataset)):
        updowns = np.array([(j['Open']-j['Close']) for j in dataset])[i-12:i]
        rs = (np.sum(updowns[updowns>=0]))/ np.abs(np.sum(updowns[updowns<0])+0.00001)
        dataset[i]['rsi_12'] = 100*rs/(1+rs)

def rsi_label(dataset):
    for i in range(1, len(dataset)-1):
        if dataset[i]['rsi_6']-dataset[i]['rsi_12'] >=0 and dataset[i+1]['rsi_6']-dataset[i]['rsi_12'] <0:
            dataset[i]['rsi_label'] = 1
        elif dataset[i]['rsi_6']-dataset[i]['rsi_12'] <=0 and dataset[i+1]['rsi_6']-dataset[i+1]['rsi_12'] >0:
            dataset[i]['rsi_label'] = -1
        else:
            dataset[i]['rsi_label'] = 0

def ema(dataset):
    ema_12 = []
    ema_26 = []
    diff = []
    macd_9 = []
    p = []
    for i in range(1, 12):
        p = np.array([j['Close'] for j in dataset[:i]])
        alpha = np.array([pow(1-2/13, j) for j in range(i)])
        ema_12.append(np.dot(p, alpha)/np.sum(alpha))
    for i in range(12, len(dataset)):
        p = np.array([j['Close'] for j in dataset[i-12:i]])
        alpha = np.array([pow(1-2/13, j) for j in range(12)])
        ema_12.append(np.dot(p, alpha)/np.sum(alpha))
    for i in range(1, 26):
        p = np.array([j['Close'] for j in dataset[:i]])
        alpha = np.array([pow(1-2/27, j) for j in range(i)])
        ema_26.append(np.dot(p, alpha)/np.sum(alpha))
    for i in range(26, len(dataset)):
        p = np.array([j['Close'] for j in dataset[i-26:i]])
        alpha = np.array([pow(1-2/27, j) for j in range(26)])
        ema_26.append(np.dot(p, alpha)/np.sum(alpha))
    diff = np.array(ema_12) - np.array(ema_26)

    for i in range(1, len(dataset)-1):
        dataset[i]['dif'] = diff[i]
    print(dataset[20])

    p = []
    for i in range(1, 9):
        p = np.array(diff[:i])
        alpha = np.array([pow(1-2/10, j) for j in range(i)])
        macd_9.append(np.dot(p, alpha)/np.sum(alpha))
    for i in range(9, len(dataset)):
        p = np.array(diff[i-9:i])
        alpha = np.array([pow(1-2/10, j) for j in range(9)])
        macd_9.append(np.dot(p, alpha)/np.sum(alpha))
    for i in range(1, len(dataset)-1):
        dataset[i]['macd_9'] = macd_9[i]

def bias_60(data, dataset):
    for i in range(2, len(dataset)-1):
        dataset[data['Date'][i+1]]['bias_60'] = ((data['Close'][i]-dataset[data['Date'][i]]['ma_60'])/(dataset[data['Date'][i]]['ma_60']))*100
    return dataset

def bias_10(data, dataset):
    for i in range(2, len(dataset)-1):
        dataset[data['Date'][i+1]]['bias_10'] = ((data['Close'][i]-dataset[data['Date'][i]]['ma_10'])/(dataset[data['Date'][i]]['ma_10']))*100
    return dataset

def bias_25(data, dataset):
    for i in range(2, len(dataset)-1):
        dataset[data['Date'][i+1]]['bias_25'] = ((data['Close'][i]-dataset[data['Date'][i]]['ma_25'])/(dataset[data['Date'][i]]['ma_25']))*100
    return dataset

def bias_10_label(data, dataset):
    for i in range(2,len(dataset)-1):
        if ((data['Close'][i]-dataset[data['Date'][i]]['ma_10'])/(dataset[data['Date'][i]]['ma_10']))*100 <= -2 and ((data['Close'][i]-dataset[data['Date'][i]]['ma_10'])/(dataset[data['Date'][i]]['ma_10']))*100 >= -3.5:
            dataset[data['Date'][i]]['bias_10_label'] = 1
        elif ((data['Close'][i]-dataset[data['Date'][i]]['ma_10'])/(dataset[data['Date'][i]]['ma_10']))*100 < -3.5:
            dataset[data['Date'][i]]['bias_10_label'] = 2
        elif ((data['Close'][i]-dataset[data['Date'][i]]['ma_10'])/(dataset[data['Date'][i]]['ma_10']))*100 >= 3 and ((data['Close'][i]-dataset[data['Date'][i]]['ma_10'])/(dataset[data['Date'][i]]['ma_10']))*100 < 5 :
            dataset[data['Date'][i]]['bias_10_label'] = -1
        elif ((data['Close'][i]-dataset[data['Date'][i]]['ma_10'])/(dataset[data['Date'][i]]['ma_10']))*100 >= 5:
            dataset[data['Date'][i]]['bias_10_label'] = -2
        else:
            dataset[data['Date'][i]]['bias_10_label']  = 0
    return dataset

'''
def bias_10_label(dataset):
    for i in range(1,len(dataset)):
        if ((dataset[i]['Close']-dataset[i]['ma_10'])/(dataset[i]['ma_10']))*100 <= -2 and ((dataset[i]['Close']-dataset[i]['ma_10'])/(dataset[i]['ma_10']))*100 >= -3.5:
            dataset[i]['bias_10_label'] = 1
        elif ((dataset[i]['Close']-dataset[i]['ma_10'])/(dataset[i]['ma_10']))*100 < -3.5:
            dataset[i]['bias_10_label'] = 2
        elif ((dataset[i]['Close']-dataset[i]['ma_10'])/(dataset[i]['ma_10']))*100 >= 3 and ((dataset[i]['Close']-dataset[i]['ma_10'])/(dataset[i]['ma_10']))*100 < 5 :
            dataset[i]['bias_10_label'] = -1
        elif ((dataset[i]['Close']-dataset[i]['ma_10'])/(dataset[i]['ma_10']))*100 >= 5:
            dataset[i]['bias_10_label'] = -2
        else:
            dataset[i]['bias_10_label'] = 0
'''
def bias_25_label(dataset):
    for i in range(1,len(dataset)):
        if ((dataset[i]['Close']-dataset[i]['ma_25'])/(dataset[i]['ma_25']))*100 <= -5 and ((dataset[i]['Close']-dataset[i]['ma_25'])/(dataset[i]['ma_25']))*100 >= -3:
            dataset[i]['bias_25_label'] = 1
        elif ((dataset[i]['Close']-dataset[i]['ma_25'])/(dataset[i]['ma_25']))*100 < -5:
            dataset[i]['bias_25_label'] = 2
        elif ((dataset[i]['Close']-dataset[i]['ma_25'])/(dataset[i]['ma_25']))*100 >= 5 and ((dataset[i]['Close']-dataset[i]['ma_25'])/(dataset[i]['ma_25']))*100 < 7 :
            dataset[i]['bias_25_label'] = -1
        elif ((dataset[i]['Close']-dataset[i]['ma_25'])/(dataset[i]['ma_25']))*100 >= 7:
            dataset[i]['bias_25_label'] = -2
        else:
            dataset[i]['bias_25_label'] = 0

def mtm_6(dataset):
    for i in range(0, 6):
        dataset[i]['mtm_6'] = dataset[i]['Close'] - dataset[0]['Close']
    for i in range(6, len(dataset)):
        dataset[i]['mtm_6'] = dataset[i]['Close'] - dataset[i-6]['Close']

def mtm_10(dataset):
    for i in range(0, 10):
        dataset[i]['mtm_10'] = dataset[i]['Close'] - dataset[0]['Close']
    for i in range(10, len(dataset)):
        dataset[i]['mtm_10'] = dataset[i]['Close'] - dataset[i-10]['Close']

def cdp(dataset):
    for i in range(len(dataset)):
        dataset[i]['cdp'] = (dataset[i]['High'] + dataset[i]['Low'] + dataset[i]['Close']*2)/4
        dataset[i]['ah'] = dataset[i]['cdp'] + dataset[i]['High'] - dataset[i]['Low']
        dataset[i]['nh'] = dataset[i]['cdp']*2 - dataset[i]['Low']
        dataset[i]['nl'] = dataset[i]['cdp']*2 - dataset[i]['High']
        dataset[i]['al'] = dataset[i]['cdp'] - dataset[i]['High'] + dataset[i]['Low']

def dmi(dataset):
    for i in range(1, len(dataset)):
        dataset[i]['dm_p'] = dataset[i]['High'] - dataset[i-1]['Low']
        dataset[i]['dm_n'] = abs(dataset[i]['Low'] - dataset[i-1]['Low'])
        dataset[i]['high_low'] = dataset[i]['High'] - dataset[i]['Low']
        dataset[i]['tr'] = max(dataset[i]['dm_p'], dataset[i]['dm_n'], dataset[i]['high_low'])

def dm_avg(dataset):
    for i in range(1, 10):
        dataset[i]['dm_p_10'] = np.mean(np.array([dataset[i]['dm_p'] for j in dataset[:i]]))
    for i in range(10, len(dataset)):
        dataset[i]['dm_p_10'] = np.mean(np.array([dataset[i]['dm_p'] for j in dataset[i-10:i]]))
    for i in range(1, 12):
        dataset[i]['dm_p_12'] = np.mean(np.array([dataset[i]['dm_p'] for j in dataset[:i]]))
    for i in range(12, len(dataset)):
        dataset[i]['dm_p_12'] = np.mean(np.array([dataset[i]['dm_p'] for j in dataset[i-12:i]]))
    for i in range(1, 14):
        dataset[i]['dm_p_14'] = np.mean(np.array([dataset[i]['dm_p'] for j in dataset[:i]]))
    for i in range(14, len(dataset)):
        dataset[i]['dm_p_14'] = np.mean(np.array([dataset[i]['dm_p'] for j in dataset[i-14:i]]))

    for i in range(1, 10):
        dataset[i]['dm_n_10'] = np.mean(np.array([dataset[i]['dm_n'] for j in dataset[:i]]))
    for i in range(10, len(dataset)):
        dataset[i]['dm_n_10'] = np.mean(np.array([dataset[i]['dm_n'] for j in dataset[i-10:i]]))
    for i in range(1, 12):
        dataset[i]['dm_n_12'] = np.mean(np.array([dataset[i]['dm_n'] for j in dataset[:i]]))
    for i in range(12, len(dataset)):
        dataset[i]['dm_n_12'] = np.mean(np.array([dataset[i]['dm_n'] for j in dataset[i-12:i]]))
    for i in range(1, 14):
        dataset[i]['dm_n_14'] = np.mean(np.array([dataset[i]['dm_n'] for j in dataset[:i]]))
    for i in range(14, len(dataset)):
        dataset[i]['dm_n_14'] = np.mean(np.array([dataset[i]['dm_n'] for j in dataset[i-14:i]]))

def tr_avg(dataset):
    for i in range(1, 10):
        dataset[i]['tr_10'] = np.mean(np.array([dataset[i]['tr'] for j in dataset[:i]]))
    for i in range(10, len(dataset)):
        dataset[i]['tr_10'] = np.mean(np.array([dataset[i]['tr'] for j in dataset[i-10:i]]))
    for i in range(1, 12):
        dataset[i]['tr_12'] = np.mean(np.array([dataset[i]['tr'] for j in dataset[:i]]))
    for i in range(12, len(dataset)):
        dataset[i]['tr_12'] = np.mean(np.array([dataset[i]['tr'] for j in dataset[i-12:i]]))
    for i in range(1, 14):
        dataset[i]['tr_14'] = np.mean(np.array([dataset[i]['tr'] for j in dataset[:i]]))
    for i in range(14, len(dataset)):
        dataset[i]['tr_14'] = np.mean(np.array([dataset[i]['tr'] for j in dataset[i-14:i]]))

def di_avg(dataset):
    for i in range(1, len(dataset)):
        dataset[i]['di_p_10'] = dataset[i]['dm_p_10'] / dataset[i]['tr_10']
        dataset[i]['di_p_12'] = dataset[i]['dm_p_12'] / dataset[i]['tr_12']
        dataset[i]['di_p_14'] = dataset[i]['dm_p_14'] / dataset[i]['tr_14']
        dataset[i]['di_n_10'] = dataset[i]['dm_n_10'] / dataset[i]['tr_10']
        dataset[i]['di_n_12'] = dataset[i]['dm_n_12'] / dataset[i]['tr_12']
        dataset[i]['di_n_14'] = dataset[i]['dm_n_14'] / dataset[i]['tr_14']

def psy_5(dataset):
    updowns = []
    for i in range(1, 5):
        updowns = np.array([(j['Open']-j['Close']) for j in dataset[:i]])
        updowns[updowns>=0] = 1
        updowns[updowns<0]  = 0
        dataset[i]['psy_5'] = (np.sum(updowns[updowns>=0]))/i
    for i in range(5, len(dataset)):
        updowns = np.array([(j['Open']-j['Close']) for j in dataset])[i-5:i]
        updowns[updowns>=0] = 1
        updowns[updowns<0]  = 0
        dataset[i]['psy_5'] = (np.sum(updowns[updowns>=0]))/5

def psy_10(dataset):
    updowns = []
    for i in range(1, 10):
        updowns = np.array([(j['Open']-j['Close']) for j in dataset[:i]])
        updowns[updowns>=0] = 1
        updowns[updowns<0]  = 0
        dataset[i]['psy_10'] = (np.sum(updowns[updowns>=0]))/i
    for i in range(10, len(dataset)):
        updowns = np.array([(j['Open']-j['Close']) for j in dataset])[i-10:i]
        updowns[updowns>=0] = 1
        updowns[updowns<0]  = 0
        dataset[i]['psy_10'] = (np.sum(updowns[updowns>=0]))/10

def psy_20(dataset):
    updowns = []
    for i in range(1, 20):
        updowns = np.array([(j['Open']-j['Close']) for j in dataset[:i]])
        updowns[updowns>=0] = 1
        updowns[updowns<0]  = 0
        dataset[i]['psy_20'] = (np.sum(updowns[updowns>=0]))/i
    for i in range(20, len(dataset)):
        updowns = np.array([(j['Open']-j['Close']) for j in dataset])[i-20:i]
        updowns[updowns>=0] = 1
        updowns[updowns<0]  = 0
        dataset[i]['psy_20'] = (np.sum(updowns[updowns>=0]))/20

def y_label(m_data, dataset):
    y = {}
    for key in dataset.keys():
        ind = np.where(m_data['Date'] == key)[0]
        y[key] = m_data['Close'][ind]

    for i in m_data['Date']:
        # max min index
        ind = np.argsort(y[i])[0]
        ind2 = np.argsort(y[i])[-1]

        # 跌20之前先漲60 or 當天最高點比開盤高60
        if (len(np.where(np.array(y[i][:ind])>y[i][ind]+20)[0])==0 and len(np.where(np.array(y[i][ind:])>y[i][ind]+60)[0])!=0) or (y[i][0]<=y[i][ind2]-60):
            dataset[i]['y'] = 1

        # 漲60之前先跌20
        elif len(np.where(np.array(y[i][:ind])>y[i][ind]+20)[0])!=0 and len(np.where(np.array(y[i][ind:])>y[i][ind]+60)[0])==0:
            dataset[i]['y'] = -1

        else:
            dataset[i]['y'] = 0
    return dataset

if '__main__' == __name__:

    data, m_data = load()
    dataset = dict.fromkeys(data['Date'])
    for i in dataset.keys():
        dataset[i] = {}

    dataset = basic(data, dataset)
    dataset = rsv(data, dataset)
    dataset = rsv_5(data, dataset)
    dataset = kd(data, dataset)
    dataset = ma_5(data, dataset)
    dataset = ma_6(data, dataset)
    dataset = ma_10(data, dataset)
    dataset = ma_12(data, dataset)
    dataset = ma_20(data, dataset)
    dataset = ma_25(data, dataset)
    dataset = ma_60(data, dataset)
    dataset = ma_5_20_label(data, dataset)
    dataset = ma_10_20_label(data, dataset)
    dataset = rsi_6(data, dataset)
    dataset = bias_60(data, dataset)
    dataset = bias_10(data, dataset)
    dataset = bias_25(data, dataset)
    dataset = bias_10_label(data, dataset)
    dataset = y_label(m_data, dataset)

    pickle.dump(dataset, open('tmp/data.pkl', 'wb'))
    exit()

    dataset = kd_label(data, dataset)
    kd_label(dataset)
    d_3_label(dataset)

    ma_5_10_20_label(dataset)
    rsi_6(dataset)
    rsi_12(dataset)
    #rsi_label(dataset)
    ema(dataset)
    #bias_10_label(dataset)
    #bias_25_label(dataset)
    mtm_6(dataset)
    mtm_10(dataset)
    cdp(dataset)
    dmi(dataset)
    dm_avg(dataset)
    tr_avg(dataset)
    di_avg(dataset)
    psy_5(dataset)
    psy_10(dataset)
    psy_20(dataset)

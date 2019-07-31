#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pickle
import datetime
import os
from database import stockDB
from sys import argv

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
    day_df = mydb.read_data(start_date, end_date, True)
    min_df = mydb.read_data(start_date, end_date, False)
    eval_result = mydb.exe_query("select * from trading_result where TS_ID = 1 and Date <= \"2019/06/30\"")
    row_day_data = {
        'Date': np.array(day_df['Date'], dtype=str),
        'Open': np.array(day_df['Open'], dtype=float),
        'High': np.array(day_df['High'], dtype=float),
        'Low': np.array(day_df['Low'], dtype=float),
        'Close': np.array(day_df['Close'], dtype=float),
        'Volume': np.array(day_df['Volume'], dtype=float),
    }

    # min_data為當天的分k資料
    row_min_data = {
        'Date': np.array(min_df['Date'], dtype=str),
        'Open': np.array(min_df['Open'], dtype=float),
        'High': np.array(min_df['High'], dtype=float),
        'Low': np.array(min_df['Low'], dtype=float),
        'Close': np.array(min_df['Close'], dtype=float),
    }


    min_data = {}
    for i in range(len(row_min_data['Date'])):
        if row_min_data['Date'][i] not in min_data.keys():
            min_data[row_min_data['Date'][i]] = {'Open': [],
                                                 'High': [],
                                                 'Low': [],
                                                 'Close': []
                                                }
        for key in min_data[row_min_data['Date'][i]].keys():
            min_data[row_min_data['Date'][i]][key].append(row_min_data[key][i])

    #pickle.dump(min_data, open('minute_data.pkl', 'wb'))

    return row_day_data, min_data


def basic(day_data, row_day_data):
    for i in range(1, len(row_day_data['Date'])):
        day_data[row_day_data['Date'][i]]['high'] = row_day_data['High'][i-1]
        day_data[row_day_data['Date'][i]]['low'] = row_day_data['Low'][i-1]
        day_data[row_day_data['Date'][i]]['open'] = row_day_data['Open'][i-1]
        day_data[row_day_data['Date'][i]]['close'] = row_day_data['Close'][i-1]
        day_data[row_day_data['Date'][i]]['volume'] = row_day_data['Volume'][i-1]
    day_data[row_day_data['Date'][0]]['high'] = row_day_data['High'][1]
    day_data[row_day_data['Date'][0]]['low'] = row_day_data['Low'][1]
    day_data[row_day_data['Date'][0]]['open'] = row_day_data['Open'][1]
    day_data[row_day_data['Date'][0]]['close'] = row_day_data['Close'][1]
    day_data[row_day_data['Date'][0]]['volume'] = row_day_data['Volume'][1]
    return day_data

def rsv(days, day_data, row_day_data):
    for i in range(2, days+1):
        close = row_day_data['Close'][i-1]
        open_ = row_day_data['Open'][i-1]
        min_ = np.min(row_day_data['Low'][:i-1])
        max_ = np.max(row_day_data['High'][:i-1])
        day_data[row_day_data['Date'][i]]['rsv_'+str(days)] = 100*(close-min_) / (max_-min_)
    for i in range(days+1, len(day_data)):
        close = np.array(row_day_data['Close'][i-1])
        open_ = np.array(row_day_data['Open'][i-1])
        min_ = np.min(row_day_data['Low'][i-(days+1):i-1])
        max_ = np.max(row_day_data['High'][i-(days+1):i-1])
        day_data[row_day_data['Date'][i]]['rsv_'+str(days)] = 100*(close-min_) / (max_-min_)
    day_data[row_day_data['Date'][0]]['rsv_'+str(days)] = 100*(row_day_data['Close'][0] - row_day_data['Low'][0]) / (row_day_data['High'][0] - row_day_data['Low'][0])
    return day_data


def k(days, day_data, row_day_data):
    day_data[row_day_data['Date'][0]]['k'] = 17.76
    day_data[row_day_data['Date'][1]]['k'] = 20.15
    day_data[row_day_data['Date'][2]]['k'] = 31.17
    for i in range(3, len(day_data)):
        day_data[row_day_data['Date'][i]]['k'] = (day_data[row_day_data['Date'][i-2]]['k'])*2/3 + (day_data[row_day_data['Date'][i-1]]['rsv_'+str(days)])*1/3
    return day_data

def d(days, day_data, row_day_data):
    day_data[row_day_data['Date'][0]]['d'] = 25.08
    day_data[row_day_data['Date'][1]]['d'] = 23.44
    day_data[row_day_data['Date'][2]]['d'] = 26.01
    for i in range(3, len(day_data)):
        day_data[row_day_data['Date'][i]]['d'] = (day_data[row_day_data['Date'][i-2]]['d'])*2/3 + (day_data[row_day_data['Date'][i-1]]['k'])*1/3
    return day_data

def kd_cross(day_data, row_day_data):
    for i in range(2, len(day_data)):
        if day_data[row_day_data['Date'][i-2]]['k'] <= day_data[row_day_data['Date'][i-2]]['d'] and day_data[row_day_data['Date'][i-1]]['k'] > day_data[row_day_data['Date'][i-1]]['d']:
            day_data[row_day_data['Date'][i]]['kd_cross'] = 1
        elif day_data[row_day_data['Date'][i-2]]['k'] >= day_data[row_day_data['Date'][i-2]]['d'] and day_data[row_day_data['Date'][i-1]]['k'] < day_data[row_day_data['Date'][i-1]]['d']:
            day_data[row_day_data['Date'][i]]['kd_cross'] = -1
        else:
            day_data[row_day_data['Date'][i]]['kd_cross'] = 0
    day_data[row_day_data['Date'][0]]['kd_cross'] = 0
    day_data[row_day_data['Date'][1]]['kd_cross'] = 0
    return day_data

def ma(days, day_data, row_day_data):
    for i in range(1, days+1):
        day_data[row_day_data['Date'][i]]['ma_'+str(days)] = np.mean(row_day_data['Close'][:i-1])
    for i in range(days+1, len(day_data)):
        close = row_day_data['Close'][i-1]
        day_data[row_day_data['Date'][i]]['ma_'+str(days)] = np.mean(row_day_data['Close'][i-(days+1):i-1])
    day_data[row_day_data['Date'][1]]['ma_'+str(days)] = row_day_data['Close'][0]
    day_data[row_day_data['Date'][0]]['ma_'+str(days)] = row_day_data['Close'][0]
    return day_data


#當天的close跟ma的差
def ma_diff(days, day_data, row_day_data):
    for i in range(1, len(day_data)):
        day_data[row_day_data['Date'][i]]['ma_'+str(days)+'_diff'] = row_day_data['Close'][i-1] - day_data[row_day_data['Date'][i-1]]['ma_'+str(days)]
    day_data[row_day_data['Date'][0]]['ma_'+str(days)+'_diff'] = day_data[row_day_data['Date'][1]]['ma_'+str(days)+'_diff']
    return day_data


#當天的close是跟ma的差是正or負
def ma_label(days, day_data, row_day_data):
    for i in range(0, len(day_data)):
        day_data[row_day_data['Date'][i]]['ma_'+str(days)+'_label'] = np.array(day_data[row_day_data['Date'][i]]['ma_'+str(days)+'_diff'])
        day_data[row_day_data['Date'][i]]['ma_'+str(days)+'_label'][day_data[row_day_data['Date'][i]]['ma_'+str(days)+'_label'] >=0 ] = 1
        day_data[row_day_data['Date'][i]]['ma_'+str(days)+'_label'][day_data[row_day_data['Date'][i]]['ma_'+str(days)+'_label'] <0 ] = 0
    return day_data

def ma_cross(day1, day2, day_data, row_day_data):
    day_data[row_day_data['Date'][1]]['ma_'+str(day1)+str(day2)+'_cross'] = 0
    day_data[row_day_data['Date'][2]]['ma_'+str(day1)+str(day2)+'_cross'] = 0
    for i in range(3, len(day_data)):
        if (day_data[row_day_data['Date'][i-2]]['ma_'+str(day1)]-day_data[row_day_data['Date'][i-2]]['ma_'+str(day2)]<0) and (day_data[row_day_data['Date'][i-1]]['ma_'+str(day1)]-day_data[row_day_data['Date'][i-1]]['ma_'+str(day2)]>=0):
            day_data[row_day_data['Date'][i]]['ma_'+str(day1)+'_'+str(day2)+'_cross'] = 1
        elif (day_data[row_day_data['Date'][i-2]]['ma_'+str(day1)]-day_data[row_day_data['Date'][i-2]]['ma_'+str(day2)]>0) and (day_data[row_day_data['Date'][i-1]]['ma_'+str(day1)]-day_data[row_day_data['Date'][i-1]]['ma_'+str(day2)]<=0):
            day_data[row_day_data['Date'][i]]['ma_'+str(day1)+'_'+str(day2)+'_cross'] = -1
        else:
            day_data[row_day_data['Date'][i]]['ma_'+str(day1)+'_'+str(day2)+'_cross'] = 0
    return day_data


def vol_avg(days, day_data, row_day_data):
    for i in range(1, days+1):
        day_data[row_day_data['Date'][i]]['vol_avg_'+str(days)] = np.mean(row_day_data['Volume'][:i-1])
    for i in range(days+1, len(day_data)):
        day_data[row_day_data['Date'][i]]['vol_avg_'+str(days)] = np.mean(row_day_data['Volume'][i-(days+1):i-1])
    day_data[row_day_data['Date'][0]]['vol_avg_'+str(days)] = day_data[row_day_data['Date'][2]]['vol_avg_'+str(days)]
    day_data[row_day_data['Date'][1]]['vol_avg_'+str(days)] = day_data[row_day_data['Date'][2]]['vol_avg_'+str(days)]
    return day_data

def rsi(days, day_data, row_day_data):
    updowns = []
    rs = []
    for i in range(2, days+1):
        updowns = np.array(row_day_data['Close'][:i-1] - row_day_data['Open'][:i-1])
        rs = (np.sum(updowns[updowns>=0]))/ np.abs(np.sum(updowns[updowns<0])+0.0000001)
        day_data[row_day_data['Date'][i]]['rsi_'+str(days)] = 100*rs/(1+rs)
    for i in range(days+1, len(day_data)):
        updowns = np.array(row_day_data['Close'][i-(days+1):i-1] - row_day_data['Open'][i-(days+1):i-1])
        rs = (np.sum(updowns[updowns>=0]))/ np.abs(np.sum(updowns[updowns<0])+0.0000001)
        day_data[row_day_data['Date'][i]]['rsi_'+str(days)] = 100*rs/(1+rs)
    day_data[row_day_data['Date'][0]]['rsi_'+str(days)] = day_data[row_day_data['Date'][2]]['rsi_'+str(days)]
    day_data[row_day_data['Date'][1]]['rsi_'+str(days)] = day_data[row_day_data['Date'][2]]['rsi_'+str(days)]
    return day_data


def rsi_label(day1, day2, day_data, row_day_data):
    for i in range(4, len(day_data)):
        if day_data[row_day_data['Date'][i-2]]['rsi_'+str(day1)]-day_data[row_day_data['Date'][i-2]]['rsi_'+str(day2)] >=0 and day_data[row_day_data['Date'][i-1]]['rsi_'+str(day1)]-day_data[row_day_data['Date'][i-1]]['rsi_'+str(day2)] <0:
            day_data[row_day_data['Date'][i]]['rsi_'+str(day1)+'_'+str(day2)+'_label'] = 1
        elif day_data[row_day_data['Date'][i-2]]['rsi_'+str(day1)]-day_data[row_day_data['Date'][i-2]]['rsi_'+str(day2)] <=0 and day_data[row_day_data['Date'][i-1]]['rsi_'+str(day1)]-day_data[row_day_data['Date'][i-1]]['rsi_'+str(day2)] >0:
            day_data[row_day_data['Date'][i]]['rsi_'+str(day1)+'_'+str(day2)+'_label'] = -1
        else:
            day_data[row_day_data['Date'][i]]['rsi_'+str(day1)+'_'+str(day2)+'_label'] = 0
    return day_data

def bias(days, day_data, row_day_data):
    for i in range(2, len(day_data)):
        day_data[row_day_data['Date'][i]]['bias_'+str(days)] = ((row_day_data['Close'][i-1]-day_data[row_day_data['Date'][i-1]]['ma_'+str(days)])/(day_data[row_day_data['Date'][i-1]]['ma_'+str(days)]))*100
    day_data[row_day_data['Date'][0]]['bias_'+str(days)] = day_data[row_day_data['Date'][2]]['bias_'+str(days)]
    day_data[row_day_data['Date'][1]]['bias_'+str(days)] = day_data[row_day_data['Date'][2]]['bias_'+str(days)]
    return day_data

def psy(days, day_data, row_day_data):
    updowns = []
    for i in range(2, days+1):
        updowns = np.array(row_day_data['Close'][:i-1] - row_day_data['Open'][:i-1])
        updowns[updowns>=0] = 1
        updowns[updowns<0]  = 0
        day_data[row_day_data['Date'][i]]['psy_'+str(days)] = (np.sum(updowns[updowns>=0]))/i
    for i in range(days+1, len(day_data)):
        updowns = np.array(row_day_data['Close'][i-6:i-1] - row_day_data['Open'][i-6:i-1])
        updowns[updowns>=0] = 1
        updowns[updowns<0]  = 0
        day_data[row_day_data['Date'][i]]['psy_'+str(days)] = (np.sum(updowns[updowns>=0]))/5
    day_data[row_day_data['Date'][0]]['psy_'+str(days)] = day_data[row_day_data['Date'][2]]['psy_'+str(days)]
    day_data[row_day_data['Date'][1]]['psy_'+str(days)] = day_data[row_day_data['Date'][2]]['psy_'+str(days)]
    return day_data

def mtm(days, day_data, row_day_data):
    for i in range(2, days+1):
        day_data[row_day_data['Date'][i]]['mtm_'+str(days)] = (row_day_data['Close'][i-1] - row_day_data['Close'][1])/row_day_data['Close'][1]
    for i in range(days+1, len(day_data)):
        day_data[row_day_data['Date'][i]]['mtm_'+str(days)] = (row_day_data['Close'][i-1] - row_day_data['Close'][i-7])/row_day_data['Close'][i-7]
    day_data[row_day_data['Date'][0]]['mtm_'+str(days)] = day_data[row_day_data['Date'][2]]['mtm_'+str(days)]
    day_data[row_day_data['Date'][1]]['mtm_'+str(days)] = day_data[row_day_data['Date'][2]]['mtm_'+str(days)]
    return day_data

def y(day_data, row_day_data, min_data, key_without_lastday, _max=60, _min=20):

    pos = 0
    neg = 0
    zero = 0
    for key in key_without_lastday:
        try:
            day_data[key]['y'] = 0
            _open = min_data[key]['Open'][0]
            gt = np.array(min_data[key]['High']) - _open
            st = np.array(min_data[key]['Low']) - _open
            
            #print(len(np.where(gt >= _max)[0]))

            ind_gt = np.where(gt >= 60)[0][0] if len(np.where(gt >= _max)[0]) else 301
            ind_st = np.where(st <= (-1)*20)[0][0] if len(np.where(st <= (-1)*_min)[0]) else 301
            #print(ind_gt, ind_st)

            if ind_gt < ind_st:
                day_data[key]['y'] = 1
                pos = pos+1

            ind_gt2 = np.where(gt >= 20)[0][0] if len(np.where(gt >= _min)[0]) else 301
            ind_st2 = np.where(st <= (-1)*60)[0][0] if len(np.where(st <= (-1)*_max)[0]) else 301

            if ind_st2 < ind_gt2:
                day_data[key]['y'] = -1
                neg = neg+1
        except:
            continue
    return day_data



if '__main__' == __name__:

    start_date = '1999/01/01'
    end_date =  str(datetime.date.today())
    
    #0-9點train data，表示要預測的是今天; 因為通常都是前一天的晚上train data，所以實際上要預測的日期是明天
    tomorrow = datetime.date.today().strftime('%Y-%m-%d') if int(datetime.datetime.now().time().strftime('%H')) <= 9 else (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    config = database_setting()
    mydb = stockDB(**config)
    row_day_data, min_data = load_database(mydb, start_date, end_date)
    key_without_lastday = row_day_data['Date']
    row_day_data['Date'] = list(row_day_data['Date'])
    row_day_data['Date'].append(tomorrow)
    row_day_data['Date'] =  np.array(row_day_data['Date'])
    
    day_data = dict.fromkeys(row_day_data['Date'])
    
    # 在以日期為keys的day_data下面，在創建一個空間放資料
    # p.s.當天的day_data來自於前一天的資料
    for i in day_data.keys():
        day_data[i] = {}

    day_data = basic(day_data, row_day_data)
    day_data = rsv(5, day_data, row_day_data)
    day_data = rsv(9, day_data, row_day_data)
    day_data = k(5, day_data, row_day_data)
    day_data = d(5, day_data, row_day_data)
    day_data = kd_cross(day_data, row_day_data)
    day_data = ma(5, day_data, row_day_data)
    day_data = ma(10, day_data, row_day_data)
    day_data = ma(20, day_data, row_day_data)
    day_data = ma(25, day_data, row_day_data)
    day_data = ma_diff(5, day_data, row_day_data)
    day_data = ma_diff(10, day_data, row_day_data)
    day_data = ma_diff(20, day_data, row_day_data)
    day_data = ma_label(5, day_data, row_day_data)
    day_data = ma_label(10, day_data, row_day_data)
    day_data = ma_label(20, day_data, row_day_data)
    day_data = ma_cross(5, 10, day_data, row_day_data)
    day_data = ma_cross(5, 20, day_data, row_day_data)
    day_data = vol_avg(6, day_data, row_day_data)
    day_data = vol_avg(12, day_data, row_day_data)
    day_data = rsi(6, day_data, row_day_data)
    day_data = rsi(12, day_data, row_day_data)
    day_data = rsi_label(6, 12, day_data, row_day_data)
    day_data = psy(5, day_data, row_day_data)
    day_data = psy(10, day_data, row_day_data)
    day_data = bias(5, day_data, row_day_data)
    day_data = bias(10, day_data, row_day_data)
    day_data = bias(20, day_data, row_day_data)
    day_data = bias(25, day_data, row_day_data)
    day_data = mtm(6, day_data, row_day_data)
    day_data = mtm(10, day_data, row_day_data)
    day_data = y(day_data, row_day_data, min_data, key_without_lastday)
    #print(day_data['2019-08-01'])

    pickle.dump(day_data[tomorrow], open('tomorrow.pkl', 'wb'))
    pickle.dump(day_data.pop(str(tomorrow), None), open('new.pkl', 'wb'))
    


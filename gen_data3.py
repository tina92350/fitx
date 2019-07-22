#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pickle
import datetime
import os
from database import stockDB

def database_setting():
    config = {
        'host': os.environ.get('stockdb_host'),
        'port': 3306,
        'user': os.environ.get('stockdb_user'),
        'password': os.environ.get('stockdb_passwd'),
        'db': 'fitx',
    }
    return config


def load_database(mydb):
    start_date = '2010/01/01'
    end_date =  '2019/07/02 23:49'
    day_df = mydb.read_data(start_date, end_date, True)
    min_df = mydb.read_data(start_date, end_date, False)
    
    day_data = {
        'Date': np.array(day_df['Date'], dtype=str),
        'Open': np.array(day_df['Open'], dtype=float),
        'High': np.array(day_df['High'], dtype=float),
        'Low': np.array(day_df['Low'], dtype=float),
        'Close': np.array(day_df['Close'], dtype=float),
        'Volume': np.array(day_df['Volume'], dtype=float),
    }
    
    # min_data為當天的分k資料
    min_data = {
        'Date': np.array(min_df['Date'], dtype=str),
        'Open': np.array(min_df['Open'], dtype=float),
        'High': np.array(min_df['High'], dtype=float),
        'Low': np.array(min_df['Low'], dtype=float),
        'Close': np.array(min_df['Close'], dtype=float),
    }

    
    return day_data, min_data


def basic(feature, day_data):
    for i in range(1, len(day_data['Date'])):
        feature[day_data['Date'][i]]['high'] = day_data['High'][i-1]
        feature[day_data['Date'][i]]['low'] = day_data['Low'][i-1]
        feature[day_data['Date'][i]]['open'] = day_data['Open'][i-1]
        feature[day_data['Date'][i]]['close'] = day_data['Close'][i-1]
        feature[day_data['Date'][i]]['volume'] = day_data['Volume'][i-1]
    return feature

def rsv(days, feature, day_data):
    for i in range(1, days+1):
        close = day_data['Close'][i]
        open_ = day_data['Open'][i]
        min_ = np.min(day_data['Low'][:days-1])
        max_ = np.max(day_data['High'][:days-1])
        feature[day_data['Date'][i]]['rsv_'+str(days)] = 100*(close-min_) / (max_-min_)
    for i in range(days+1, len(feature)):
        close = day_data['Close'][i]
        open_ = day_data['Open'][i]
        min_ = np.min(day_data['Low'][i-(days+1):i-1])
        max_ = np.max(day_data['High'][i-(days+1):i-1])
        feature[day_data['Date'][i]]['rsv_'+str(days)] = 100*(close-min_) / (max_-min_)
    feature[day_data['Date'][0]]['rsv_'+str(days)] = 100*(day_data['Close'][0] - day_data['Low'][0]) / (day_data['High'][0] - day_data['Low'][0])
    return feature

def ma(days, feature, day_data):
    for i in range(1, days+1):
        feature[day_data['Date'][i]]['ma_'+str(days)] = np.mean(day_data['Close'][:i-1])
    for i in range(days+1, len(feature)):
        close = day_data['Close'][i]
        feature[day_data['Date'][i]]['ma_'+str(days)] = np.mean(day_data['Close'][i-(days+1):i-1])
    feature[day_data['Date'][1]]['ma_'+str(days)] = day_data['Close'][0]
    feature[day_data['Date'][0]]['ma_'+str(days)] = day_data['Close'][0]     
    return feature


#當天的close跟ma的差
def ma_diff(days, feature, day_data):
    for i in range(1, len(feature)):
        feature[day_data['Date'][i]]['ma_'+str(days)+'_diff'] = day_data['Close'][i-1] - feature[day_data['Date'][i-1]]['ma_'+str(days)]
    feature[day_data['Date'][0]]['ma_'+str(days)+'_diff'] = feature[day_data['Date'][1]]['ma_'+str(days)+'_diff']
    return feature


#當天的close是跟ma的差是正or負
def ma_label(days, feature, day_data):
    for i in range(0, len(feature)):
        feature[day_data['Date'][i]]['ma_'+str(days)+'_label'] = np.array(feature[day_data['Date'][i]]['ma_'+str(days)+'_diff'])
        feature[day_data['Date'][i]]['ma_'+str(days)+'_label'][feature[day_data['Date'][i]]['ma_'+str(days)+'_label'] >=0 ] = 1
        feature[day_data['Date'][i]]['ma_'+str(days)+'_label'][feature[day_data['Date'][i]]['ma_'+str(days)+'_label'] <0 ] = 0
    return feature

def ma_cross(day1, day2, feature, day_data):
    feature[day_data['Date'][1]]['ma_'+str(day1)+str(day2)+'_cross'] = 0
    feature[day_data['Date'][2]]['ma_'+str(day1)+str(day2)+'_cross'] = 0
    for i in range(3, len(feature)-1):
        if (feature[day_data['Date'][i-2]]['ma_'+str(day1)]-feature[day_data['Date'][i-2]]['ma_'+str(day2)]<0) and (feature[day_data['Date'][i-1]]['ma_'+str(day1)]-feature[day_data['Date'][i-1]]['ma_'+str(day2)]>=0):
            feature[day_data['Date'][i]]['ma_'+str(day1)+'_'+str(day2)+'_cross'] = 1
        elif (feature[day_data['Date'][i-2]]['ma_'+str(day1)]-feature[day_data['Date'][i-2]]['ma_'+str(day2)]>0) and (feature[day_data['Date'][i-1]]['ma_'+str(day1)]-feature[day_data['Date'][i-1]]['ma_'+str(day2)]<=0):
            feature[day_data['Date'][i]]['ma_'+str(day1)+'_'+str(day2)+'_cross'] = -1
        else:
            feature[day_data['Date'][i]]['ma_'+str(day1)+'_'+str(day2)+'_cross'] = 0
    return feature


def vol_avg(days, feature, day_data):
    for i in range(1, days+1):
        feature[day_data['Date'][i]]['vol_avg_'+str(days)] = np.mean(day_data['Volume'][:i-1])
    for i in range(days+1, len(feature)):
        feature[day_data['Date'][i]]['vol_avg_'+str(days)] = np.mean(day_data['Volume'][i-(days+1):i-1])
    feature[day_data['Date'][0]]['vol_avg_'+str(days)] = feature[day_data['Date'][1]]['vol_avg_'+str(days)]
    return feature

def rsi(days, feature, day_data):
    updowns = []
    rs = []
    for i in range(1, days+1):
        updowns = np.array(day_data['Close'][:i] - day_data['Open'][:i])
        rs = (np.sum(updowns[updowns>=0]))/ np.abs(np.sum(updowns[updowns<0]))
        feature[day_data['Date'][i+1]]['rsi_'+str(days)] = 100*rs/(1+rs)
    for i in range(days+1, len(feature)-1):
        updowns = np.array(day_data['Close'][i-(days):i] - day_data['Open'][i-(days):i])
        rs = (np.sum(updowns[updowns>=0]))/ np.abs(np.sum(updowns[updowns<0]))
        feature[day_data['Date'][i+1]]['rsi_'+str(days)] = 100*rs/(1+rs)
    feature[day_data['Date'][0]]['rsi_'+str(days)] = feature[day_data['Date'][2]]['rsi_'+str(days)]
    feature[day_data['Date'][1]]['rsi_'+str(days)] = feature[day_data['Date'][2]]['rsi_'+str(days)]
    return feature


def rsi_label(day1, day2, feature, day_data):
    for i in range(3, len(feature)-1):
        if feature[day_data['Date'][i-1]]['rsi_'+str(day1)]-feature[day_data['Date'][i-1]]['rsi_'+str(day2)] >=0 and feature[day_data['Date'][i]]['rsi_'+str(day1)]-feature[day_data['Date'][i]]['rsi_'+str(day2)] <0:
            feature[day_data['Date'][i+1]]['rsi_'+str(day1)+'_'+str(day2)+'_label'] = 1
        elif feature[day_data['Date'][i-1]]['rsi_'+str(day1)]-feature[day_data['Date'][i-1]]['rsi_'+str(day2)] <=0 and feature[day_data['Date'][i]]['rsi_'+str(day1)]-feature[day_data['Date'][i]]['rsi_'+str(day2)] >0:
            feature[day_data['Date'][i+1]]['rsi_'+str(day1)+'_'+str(day2)+'_label'] = -1
        else:
            feature[day_data['Date'][i+1]]['rsi_'+str(day1)+'_'+str(day2)+'_label'] = 0
    return feature

def bias(days, feature, day_data):
    for i in range(1, len(feature)-1):
        feature[day_data['Date'][i+1]]['bias_'+str(days)] = ((day_data['Close'][i]-feature[day_data['Date'][i]]['ma_'+str(days)])/(feature[day_data['Date'][i]]['ma_'+str(days)]))*100
    return feature

            

def psy(days, feature, day_data):
    updowns = []
    for i in range(1, days+1):
        updowns = np.array(day_data['Close'][:i] - day_data['Open'][:i])
        updowns[updowns>=0] = 1
        updowns[updowns<0]  = 0
        feature[day_data['Date'][i+1]]['psy_'+str(days)] = (np.sum(updowns[updowns>=0]))/i
    for i in range(days+1, len(feature)-1):
        updowns = np.array(day_data['Close'][i-5:i] - day_data['Open'][i-5:i])
        updowns[updowns>=0] = 1
        updowns[updowns<0]  = 0
        feature[day_data['Date'][i+1]]['psy_'+str(days)] = (np.sum(updowns[updowns>=0]))/5
    return feature


def mtm(days, feature, day_data):
    for i in range(1, days):
        feature[day_data['Date'][i+1]]['mtm_'+str(days)] = (day_data['Close'][i] - day_data['Close'][1])/day_data['Close'][1]
    for i in range(days, len(feature)-1):
        feature[day_data['Date'][i+1]]['mtm_'+str(days)] = (day_data['Close'][i] - day_data['Close'][i-6])/day_data['Close'][i-6]
    return feature


def y(feature, day_data, min_data):
        
    for key in feature.keys():
        feature[key]['y'] = 0
        
        _open = min_data[key]['Open'][0]
        gt = np.array(min_data[key]['High']) - _open
        st = np.array(min_data[key]['Low']) - _open
        
        
        
        ind_gt = np.where(gt >= _max)[0][0] if len(np.where(gt >= _max)[0]) else 301
        ind_st = np.where(st <= (-1)*_min)[0][0] if len(np.where(st <= (-1)*_min)[0]) else 301
        
        print('pos')
        print(ind_gt, ind_st)
        
        if ind_gt < ind_st:
            feature[key]['y'] = 1
        
        ind_gt2 = np.where(gt >= _min)[0][0] if len(np.where(gt >= _min)[0]) else 301
        ind_st2 = np.where(st <= (-1)*_max)[0][0] if len(np.where(st <= (-1)*_max)[0]) else 301
        print('neg')
        print(ind_gt2, ind_st2)
        
        
        if ind_st2 < ind_gt2:
            feature[key]['y'] = -1
            
    return feature
            


if '__main__' == __name__:

    config = database_setting()
    mydb = stockDB(**config)
    day_data, min_data = load_database(mydb)
   
    key = []
    key_not_exist = []
    for i in day_data['Date']:
        if i in min_data['Date']:
            key.append(i)
        if i not in min_data['Date']:
            key_not_exist.append(i)
            
    feature = dict.fromkeys(key)
    
    print(key_not_exist)
    exit()
    
    
    
    
    # 在以日期為keys的feature下面，在創建一個空間放資料
    # p.s.當天的feature來自於前一天的資料
    for i in feature.keys():
        feature[i] = {}
    
    feature = basic(feature, day_data)
    feature = rsv(5, feature, day_data)
    feature = rsv(9, feature, day_data)
    feature = ma(5, feature, day_data)
    feature = ma(10, feature, day_data)
    feature = ma(20, feature, day_data)
    feature = ma(25, feature, day_data)
    feature = ma_diff(5, feature, day_data)
    feature = ma_diff(10, feature, day_data)
    feature = ma_diff(20, feature, day_data)
    feature = ma_label(5, feature, day_data)
    feature = ma_label(10, feature, day_data)
    feature = ma_label(20, feature, day_data)
    feature = ma_cross(5, 10, feature, day_data)
    feature = ma_cross(5, 20, feature, day_data)
    feature = vol_avg(6, feature, day_data)
    feature = vol_avg(12, feature, day_data)
    feature = rsi(6, feature, day_data)
    feature = rsi(12, feature, day_data)
    feature = rsi_label(6, 12, feature, day_data)
    feature = psy(5, feature, day_data)
    feature = psy(10, feature, day_data)
    feature = bias(5, feature, day_data)
    feature = bias(10, feature, day_data)
    feature = bias(20, feature, day_data)
    feature = bias(25, feature, day_data)
    feature = mtm(6, feature, day_data)
    feature = mtm(10, feature, day_data)
    feature = y(feature, day_data, min_data)
    



    
        
    exit()
    


       
    


     
    
    #print(feature['2010-01-05'])
    
    
    


#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pickle
import datetime
import os
from database import stockDB
from sys import argv
import pandas_datareader.data as web

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
    
    return day_df, min_df


def within_day(day_df):
    day_df['High-Low'] = day_df['High']-day_df['Low']
    day_df['Close-Open'] = day_df['Close']-day_df['Open']
    day_df['High-Open'] = day_df['High']-day_df['Open']
    day_df['High-Close'] = day_df['High']-day_df['Close']
    day_df['Open-Low'] = day_df['Open']-day_df['Low']
    day_df['Close-Low'] = day_df['Close']-day_df['Low']
    day_df.loc[day_df['Close-Open']>=0, 'Rise_or_not'] = 1
    day_df.loc[day_df['Close-Open']<0, 'Rise_or_not'] = 0
    return day_df

def ma(days, day_df):
    day_df['ma_'+str(days)] = day_df['Close'].rolling(days).mean()
    day_df['Open-ma'+str(days)] = day_df['Open'] - day_df['ma_'+str(days)]
    day_df['High-ma'+str(days)] = day_df['High'] - day_df['ma_'+str(days)]
    day_df['Low-ma'+str(days)] = day_df['Low'] - day_df['ma_'+str(days)]
    day_df['Close-ma'+str(days)] = day_df['Low'] - day_df['ma_'+str(days)]
    return day_df

def ma_cross(day1, day2, day_df):
    index = 'ma'+str(day1)+'-'+'ma'+str(day2)
    day_df[index] = day_df['ma_'+str(day1)]-day_df['ma_'+str(day2)]
    day_df.loc[day_df[index]>=0, index+'_label'] = 1
    day_df.loc[day_df[index]<0, index+'_label'] = -1
    day_df[index+'_cross'] = 0
    day_df.loc[(day_df[index]>=0) & (day_df[index].shift()<0) , index+'_cross'] = -1
    day_df.loc[(day_df[index]<=0) & (day_df[index].shift()>0) , index+'_cross'] = 1
    return day_df


def rsv(days, day_df):
    day_df['high'] = day_df['High'].rolling(days).max()
    day_df['low'] = day_df['Low'].rolling(days).min()
    day_df['rsv_'+str(days)] = 100*(day_df['Close']- day_df['low']) / (day_df['high'] - day_df['low'])
    day_df = day_df.drop(['high', 'low'] , axis=1)
    return day_df

def kd(day_df):
    
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 5) , 'k'] = 17.76
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 5) , 'd'] = 25.08
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 6) , 'k'] = 20.15
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 6) , 'd'] = 23.44
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 7) , 'k'] = 31.17
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 7), 'd'] = 26.01
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 8), 'k'] = 36.45
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 8), 'd'] = 29.49
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 11), 'k'] = 42.47
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 11), 'd'] = 33.82
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 12), 'k'] = 46.13
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 12), 'd'] = 37.92
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 13), 'k'] = 46.78
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 13), 'd'] = 40.87
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 14), 'k'] = 44.59
    day_df.loc[day_df['Date']==datetime.date(1999, 1, 14), 'd'] = 42.11
    #day_df.loc[day_df['Date']==datetime.date(1999, 1, 15), 'k'] = 63.06
    #day_df.loc[day_df['Date']==datetime.date(1999, 1, 15), 'd'] = 42.11
    
    with_k = [datetime.date(1999, 1, 5), datetime.date(1999, 1, 6), datetime.date(1999, 1, 7), datetime.date(1999, 1, 8), datetime.date(1999, 1, 11), datetime.date(1999, 1, 12), datetime.date(1999, 1, 13), datetime.date(1999, 1, 14), datetime.date(1999, 1, 15)]
    
    without_k = list(day_df['Date'])
    
    for i in with_k:
        if i in key:
            without_k.remove(i)

    for i in range(1, len(key)):
        day_df.loc[day_df['Date']==datetime.date(1999, 1, 15), 'k']= 63.06
        day_df.loc[day_df['Date']==key[i], 'k'] = day_df.loc[day_df['Date']==key[i-1], 'k']*2/3 + day_df.loc[day_df['Date']==key[i-1], 'rsv_9']*1/3
    
    return day_df
    
    
def rsi(days, day_df):
    day_df.loc[day_df['Close-Open']>=0, 'pos'] = day_df['Close-Open']
    day_df.loc[day_df['Close-Open']<0, 'pos'] = 0
    day_df.loc[day_df['Close-Open']<=0, 'neg'] = abs(day_df['Close-Open'])
    day_df.loc[day_df['Close-Open']>0, 'neg'] = 0
    day_df['rsi_'+str(days)] = 100*day_df['pos'].rolling(days).sum() / (day_df['pos'].rolling(days).sum()+day_df['neg'].rolling(days).sum())
    day_df = day_df.drop(['pos', 'neg'], axis=1)
    return day_df

def rsi_cross(day1, day2, day_df):
    index = 'rsi'+str(day1)+'-'+'rsi'+str(day2)
    day_df[index] = day_df['rsi_'+str(day1)]-day_df['rsi_'+str(day2)]
    day_df[index+'_cross'] = 0
    day_df.loc[(day_df[index]>=0) & (day_df[index].shift()<0) , index+'_cross'] = -1
    day_df.loc[(day_df[index]<=0) & (day_df[index].shift()>0) , index+'_cross'] = 1
    return day_df

def ema(day_df):
    day_df['ema_12'] = pd.Series.ewm(day_df['Close'], span=12).mean()
    day_df['ema_26'] = pd.Series.ewm(day_df['Close'], span=26).mean()
    day_df['dif'] = day_df['ema_12'] - day_df['ema_26']
    day_df['macd'] = pd.Series.ewm(day_df['dif'], span=9).mean()
    day_df['dif-macd'] = day_df['dif'] - day_df['macd']
    day_df['macd_index'] = 0
    day_df.loc[(day_df['dif-macd']>=0) & (day_df['dif-macd'].shift()<0) , 'macd_index'] = 1
    day_df.loc[(day_df['dif-macd']<0) & (day_df['dif-macd'].shift()>0) , 'macd_index'] = -1
    return day_df
    

def vol(days, day_df):
    day_df['vol_avg_'+str(days)] = day_df['Volume'].rolling(days).mean()
    day_df['vol_dif_'+str(days)] = day_df['Volume'] - day_df['Volume'].rolling(days).mean()
    day_df['vol_dif_label'+str(days)] = 0
    day_df.loc[day_df['vol_dif_'+str(days)]>=0, 'vol_dif_label'+str(days)] = 1
    day_df.loc[day_df['vol_dif_'+str(days)]<=0, 'vol_dif_label'+str(days)] = -1
    return day_df

def mtm(days, day_df):
    day_df['mtm_'+str(days)] = day_df['Close'] - day_df['Close'].shift(days)
    return day_df

def psy(days, day_df):
    day_df['psy_'+str(days)] = day_df['Rise_or_not'].rolling(days).sum()
    return day_df

def nasdaq(day_df, start_date, end_date):
    nasdaq = web.DataReader("^IXIC", "yahoo", start=start_date, end=end_date)
    a = []
    for i in range(len(nasdaq)):
        a.append((nasdaq.index[i]).to_pydatetime().date())
    nasdaq['Date'] = a
    nasdaq['close_open'] = nasdaq['Close'] - nasdaq['Open']
    nasdaq['change_rate'] = 100*(nasdaq['Close'] - nasdaq['Open'])/ nasdaq['Open']
    nasdaq['nasdaq']= nasdaq['Close'] - nasdaq['Open']
    nasdaq.loc[nasdaq.close_open>=0, 'nasdaq'] = 1
    nasdaq.loc[nasdaq.close_open<0, 'nasdaq'] = 0
    for date in day_df['Date']:
        if date in nasdaq['Date']:
            day_df.loc[day_df['Date']==date, 'nasdaq'] = nasdaq.loc[date, 'nasdaq']
            day_df.loc[day_df['Date']==date, 'nasdaq_point'] = nasdaq.loc[date, 'close_open']
            day_df.loc[day_df['Date']==date, 'nasdaq_rate'] = nasdaq.loc[date, 'change_rate']
    return day_df

def dji(day_df, start_date, end_date):
    dji = web.DataReader("^DJI", "yahoo", start=start_date, end=end_date)
    a = []
    for i in range(len(dji)):
        a.append((dji.index[i]).to_pydatetime().date())
    dji['Date'] = a
    dji['close_open'] = dji['Close'] - dji['Open']
    dji['change_rate'] = 100*(dji['Close'] - dji['Open'])/ dji['Open']
    dji['dji']= dji['Close'] - dji['Open']
    dji.loc[dji.close_open>=0, 'dji'] = 1
    dji.loc[dji.close_open<0, 'dji'] = 0
    for date in day_df['Date']:
        if date in dji['Date']:
            day_df.loc[day_df['Date']==date, 'dji'] = dji.loc[date, 'dji']
            day_df.loc[day_df['Date']==date, 'dji_point'] = dji.loc[date, 'close_open']
            day_df.loc[day_df['Date']==date, 'dji_rate'] = dji.loc[date, 'change_rate']
    return day_df


def sp500(day_df, start_date, end_date):
    sp500 = web.DataReader("^GSPC", "yahoo", start=start_date, end=end_date)
    a = []
    for i in range(len(sp500)):
        a.append((sp500.index[i]).to_pydatetime().date())
    sp500['Date'] = a
    sp500['close_open'] = sp500['Close'] - sp500['Open']
    sp500['change_rate'] = 100*(sp500['Close'] - sp500['Open'])/ sp500['Open']
    sp500['sp500']= sp500['Close'] - sp500['Open']
    sp500.loc[sp500.close_open>=0, 'sp500'] = 1
    sp500.loc[sp500.close_open<0, 'sp500'] = 0
    for date in day_df['Date']:
        if date in sp500['Date']:
            day_df.loc[day_df['Date']==date, 'sp500'] = sp500.loc[date, 'sp500']
            day_df.loc[day_df['Date']==date, 'sp500_point'] = sp500.loc[date, 'close_open']
            day_df.loc[day_df['Date']==date, 'sp500_rate'] = sp500.loc[date, 'change_rate']
    return day_df


def moex(day_df, start_date, end_date):
    moex = web.DataReader("IMOEX.ME", "yahoo", start=start_date, end=end_date)
    a = []
    for i in range(len(moex)):
        a.append((moex.index[i]).to_pydatetime().date())
    moex['Date'] = a
    moex['close_open'] = moex['Close'] - moex['Open']
    moex['change_rate'] = 100*(moex['Close'] - moex['Open'])/ moex['Open']
    moex['moex']= moex['Close'] - moex['Open']
    moex.loc[moex.close_open>=0, 'moex'] = 1
    moex.loc[moex.close_open<0, 'moex'] = 0
    for date in day_df['Date']:
        if date in moex['Date']:
            day_df.loc[day_df['Date']==date, 'moex'] = moex.loc[date, 'moex']
            day_df.loc[day_df['Date']==date, 'moex_point'] = moex.loc[date, 'close_open']
            day_df.loc[day_df['Date']==date, 'moex_rate'] = moex.loc[date, 'change_rate']
    return day_df

def cac40(day_df, start_date, end_date):
    cac40 = web.DataReader("^FCHI", "yahoo", start=start_date, end=end_date)
    a = []
    for i in range(len(cac40)):
        a.append((cac40.index[i]).to_pydatetime().date())
    cac40['Date'] = a
    cac40['close_open'] = cac40['Close'] - cac40['Open']
    cac40['change_rate'] = 100*(cac40['Close'] - cac40['Open'])/ cac40['Open']
    cac40['cac40']= cac40['Close'] - cac40['Open']
    cac40.loc[cac40.close_open>=0, 'cac40'] = 1
    cac40.loc[cac40.close_open<0, 'cac40'] = 0
    for date in day_df['Date']:
        if date in cac40['Date']:
            day_df.loc[day_df['Date']==date, 'cac40'] = cac40.loc[date, 'cac40']
            day_df.loc[day_df['Date']==date, 'cac40_point'] = cac40.loc[date, 'close_open']
            day_df.loc[day_df['Date']==date, 'cac40_rate'] = cac40.loc[date, 'change_rate']
    return day_df

def russell(day_df, start_date, end_date):
    russell = web.DataReader("^RUT", "yahoo", start=start_date, end=end_date)
    a = []
    for i in range(len(russell)):
        a.append((russell.index[i]).to_pydatetime().date())
    russell['Date'] = a
    russell['close_open'] = russell['Close'] - russell['Open']
    russell['change_rate'] = 100*(russell['Close'] - russell['Open'])/ russell['Open']
    russell['russell']= russell['Close'] - russell['Open']
    russell.loc[russell.close_open>=0, 'russell'] = 1
    russell.loc[russell.close_open<0, 'russell'] = 0
    for date in day_df['Date']:
        if date in russell['Date']:
            day_df.loc[day_df['Date']==date, 'russell'] = russell.loc[date, 'russell']
            day_df.loc[day_df['Date']==date, 'russell_point'] = russell.loc[date, 'close_open']
            day_df.loc[day_df['Date']==date, 'russell_rate'] = russell.loc[date, 'change_rate']
    return day_df

def y_today(day_df, min_df, _max, _min):
    date = []   
    for i in min_df['Date']:
        if i not in date:
            date.append(i)
    a = min_df.groupby('Date')
    pos=0
    neg=0
    day_df['y_today'] = 0
    for i in date:
        b = a.get_group(i)
        _open = b['Open'].head(1)
        high_open = np.array(b["High"]) - np.array(_open)
        low_open = np.array(b['Low']) - np.array(_open)
        ind1 = np.where(high_open>=_max)[0][0] if len(np.where(high_open >= _max)[0]) else 301
        ind2 = np.where(low_open<=(-1)*_min)[0][0] if len(np.where(low_open <= (-1)*_min)[0]) else 301
        #print(ind1, ind2)
        if ind1 < ind2:
            day_df.loc[day_df['Date'] ==i, 'y_today'] = 1
            pos = pos+1
        ind3 = np.where(high_open >= _min)[0][0] if len(np.where(high_open >= _min)[0]) else 301
        ind4 = np.where(low_open <= (-1)*_max)[0][0] if len(np.where(low_open <= (-1)*_max)[0]) else 301
        if ind3 > ind4:
            day_df.loc[day_df['Date'] ==i, 'y_today'] = -1
            neg = neg+1
    print(pos, neg, len(date)-pos-neg)
    return day_df


def y_to_predict(day_df):
    day_df['y_to_predict'] = day_df['y_today'].shift(-1)
    return day_df
    

if '__main__' == __name__:

    start_date = '1999/01/01'
    end_date =  datetime.datetime.now()
    end_date2 = datetime.datetime.now().date()
    _max=60
    _min=20
    config = database_setting()
    
    mydb = stockDB(**config)
    day_df, min_df = load_database(mydb, start_date, end_date)
    
    df_end_date = datetime.datetime.strptime(str(day_df.iloc[-1]['Date']), '%Y-%m-%d').date()

    if df_end_date != end_date.date() and   df_end_date != end_date.date()+datetime.timedelta(days=-1):
        print('+++ there is sth wrong with database, plz check +++')
    
    day_df = within_day(day_df)
    day_df = ema(day_df)
    day_df = ma(5, day_df)
    day_df = ma(10, day_df)
    day_df = ma(20, day_df)
    day_df = ma(60, day_df)
    day_df = ma_cross(5, 10, day_df)
    day_df = ma_cross(5, 20, day_df)
    day_df = ma_cross(10, 20, day_df)
    day_df = rsv(5, day_df)
    day_df = rsv(9, day_df)
    day_df = rsi(6, day_df)
    day_df = rsi(12, day_df)
    day_df = rsi_cross(6, 12, day_df)
    day_df = vol(3, day_df)
    day_df = vol(6, day_df)
    day_df = vol(12, day_df)
    day_df = mtm(3, day_df)
    day_df = mtm(5, day_df)
    day_df = mtm(6, day_df)
    day_df = mtm(10, day_df)
    day_df = psy(3, day_df)
    day_df = psy(5, day_df)
    day_df = psy(10, day_df)
    day_df = psy(20, day_df)
    day_df = nasdaq(day_df, start_date, end_date)
    day_df = dji(day_df, start_date, end_date)
    day_df = sp500(day_df, start_date, end_date)
    day_df = moex(day_df, start_date, end_date2)
    day_df = cac40(day_df, start_date, end_date)
    day_df = russell(day_df, start_date, end_date)
    day_df = y_today(day_df, min_df, _max, _min)
    day_df = y_to_predict(day_df)

    day_df = day_df.fillna(0)
    
    pickle.dump(day_df, open('./df.pkl', 'wb'))

    print(day_df.columns)
    print(len(day_df.columns))

    

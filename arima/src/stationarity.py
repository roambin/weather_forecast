import pandas as pd
import matplotlib.pylab as plt
#2.python判断时序数据稳定性
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #这里以一年为一个窗口，每一个时间t的值由它前面12个月（包括自己）的均值代替，标准差同理。
    rolmean = timeseries.rolling(12).mean()#pd.rolling_mean(timeseries,window=12)
    rolstd = timeseries.rolling(12).std()#pd.rolling_std(timeseries, window=12)
    
    #plot rolling statistics:
    fig = plt.figure()
    fig.add_subplot()
    orig = plt.plot(timeseries, color = 'blue',label='Original')
    mean = plt.plot(rolmean , color = 'red',label = 'rolling mean')
    std = plt.plot(rolstd, color = 'black', label= 'Rolling standard deviation')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation')
    #plt.show(block=False)
    
    #Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries,autolag = 'AIC')
    #dftest的输出前一项依次为检测值，p值，滞后数，使用的观测数，各个置信度下的临界值
    dfoutput = pd.Series(dftest[0:4],index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical value (%s)' %key] = value
    
    print (dfoutput)
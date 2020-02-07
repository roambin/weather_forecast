import numpy as np  
import matplotlib.pyplot as plt

def dataHandle(dataset):
    blankIndex=[]
    blankValue=[]
    x=[]
    y=[]
    #delete unknown points at begin and end
    for i in range(len(dataset)):
        if int(dataset[i,0])==-99:
            dataset=np.delete(dataset,i,0)
        else:
            break
    for i in range(len(dataset)):
        if int(dataset[len(dataset)-i-1,0])==-99:
            dataset=np.delete(dataset,len(dataset)-i-1,0)
        else:
            break
    #set list
    for i in range(len(dataset)):
        if int(dataset[i,0])==-99:
            blankIndex.append(i)
        else:
            x.append(i)
            y.append(dataset[i,0])
    # 生成多项式对象
    coefficient = np.polyfit(x, y, int(len(x)*0.1))
    func = np.poly1d(coefficient)
    # predict blank value
    for j in range(len(blankIndex)):
        dataset[blankIndex[j],0]=func(blankIndex[j])
        blankValue.append(dataset[blankIndex[j],0])
    #plot
    fig=plt.figure("fitting")
    plt.plot(x,y,color='#0B649F',linestyle='',marker='.',label='input point')
    plt.plot(blankIndex,blankValue,color='#ff0066',linestyle='',marker='x',label='fitting point')
    plt.plot(range(len(dataset)),dataset[:,0].tolist(),color='g',linestyle='-',label='fitting line') 
    plt.legend(loc='upper left')
    return dataset,plt,func,fig
def predict(func,dataset):
    predictLen=100
    prepLen=len(dataset)
    predictRes=[]
    for i in range(predictLen):
        predictRes.append(func(prepLen+i))
    fig=plt.figure("fitting predict")
    plt.plot(range(predictLen),predictRes,color='#ff0066',linestyle='-',marker='.',label='predict')
    plt.legend(loc='upper left')
    return plt,fig

'''
#test
from pandas import read_csv
import os
placelist=['Global','Northern hemisphere','Southern hemisphere','Suzhou','Hawaii','Actic','Xinjiang']
placename=placelist[5]
filename='data'+os.sep+placename+'.csv'
dataframe = read_csv(filename, usecols=[1], engine='python', skipfooter=0)
dataset = dataframe.values
dataset = dataset.astype('float32')
dataset,fittingPlot,func,fittingFig=dataHandle(dataset)
fittingPredictPlot,fittingPredictFig=predict(func,dataset)
fittingPlot.show()
fittingPredictPlot.show()
'''
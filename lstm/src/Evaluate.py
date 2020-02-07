from numpy import mean,diff

def evaluate(rawData):
    #data
    data=rawData[:,0]
    dataMin=data.min()
    dataMax=data.max()
    dataAvg=mean(data)
    #difference of data
    diffOrder=1
    diffData=diff(data,n=diffOrder)
    diffMin=diffData.min()
    diffMax=diffData.max()
    diffAvg=mean(diffData)
    evaluate=[dataMin,dataAvg,dataMax,diffMin,diffAvg,diffMax]
    name=['min','avg','max','diff min','diff avg','diff max']
    for i in range(len(evaluate)):
        print(name[i]+":"+str(evaluate[i]))
'''
#test
from pandas import read_csv
filename='data.csv'
dataframe = read_csv(filename, usecols=[1], engine='python', skipfooter=0)
dataset = dataframe.values
dataset = dataset.astype('float32')
evaluate(dataset)
'''
# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import initializers
from keras import optimizers
import Fitting
from Fitting import dataHandle
from Evaluate import evaluate
import os
import shutil
from Log import Logger
import sys
# convert an array of values into a dataset matrix
def create_dataset(data, look_back):
    dataX, dataY = [], []
    for i in range(len(data)-look_back):
        dataX.append(data[i:(i+look_back), 0])
        dataY.append(data[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
# calculate root mean squared error
def rmse(real,predict,name):
    score = math.sqrt(mean_squared_error(real, predict))
    print(name+' Score: %.2f RMSE' % (score))
# shift data for plotting
def setPlot(data,length,begin,end):
    plot = numpy.empty([length,1])
    plot[:, :] = numpy.nan
    plot[begin:end, :] = data
    return  plot
# make next predictions
def nextPredict(model,data,length):
    predict=[]
    testElem=data
    for i in range(length):
        i
        modelRes=model.predict(testElem)[0][0]
        predict.append([modelRes])
        testElem[0][0][0:look_back-1]=testElem[0][0][1:look_back]
        testElem[0][0][look_back-1]=modelRes
    return predict
# create model
def createModel():
    model = Sequential()
    #model.add(LSTM(128, input_shape=(1, look_back)))
    #model.add(Dense(1))
    model.add(LSTM(35, input_shape=(1, look_back), return_sequences=True, kernel_initializer='orthogonal', bias_initializer='zeros'))
    model.add(LSTM(10, return_sequences=True, kernel_initializer='orthogonal', bias_initializer='zeros'))
    model.add(LSTM(3, return_sequences=True, kernel_initializer='orthogonal', bias_initializer='zeros'))
    model.add(LSTM(7, return_sequences=False, kernel_initializer='orthogonal', bias_initializer='zeros'))
    model.add(Dense(units=1, activation='tanh', kernel_initializer=initializers.random_normal(mean=0.02,stddev=0.05), bias_initializer='zeros'))
    adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, clipnorm=15)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
placelist=['Global','Northern hemisphere','Southern hemisphere','Suzhou','Hawaii','Arctic','Xinjiang']
placename=placelist[3]
filename='data'+os.sep+placename+'.csv'
dataframe = read_csv(filename, usecols=[1], engine='python', skipfooter=0)
dataset = dataframe.values
dataset = dataset.astype('float32')
dataset,fittingPlot,func,fittingFig=dataHandle(dataset)
fittingPredictPlot,fittingPredictFig=Fitting.predict(func,dataset)
#create file
if os.path.exists('result')==False:
    os.mkdir('result')
path='result'+os.sep+placename
if os.path.exists(path):
    shutil.rmtree(path)
os.mkdir(path)
#save log
sys.stdout = Logger(path+os.sep+placename+'.txt')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# set train and test
look_back = 40
train_size = int(len(dataset) * 1)
train=dataset[0:train_size,:]
test_size = look_back
test = dataset[0:test_size,:]
future_size = look_back
future = dataset[len(dataset)-future_size:len(dataset),:]
# reshape into X=t and Y=t+1
trainX, trainY = create_dataset(train, look_back)
testX=numpy.array([[test[:,0]]])
futureX=numpy.array([[future[:,0]]])
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# create and fit the LSTM network
epochs=180
testLen=int(len(trainX)*0.8)
modelTest=createModel()
model=createModel()
modelTest.fit(trainX[:testLen], trainY[:testLen], epochs=epochs, batch_size=1, verbose=2)
model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)
# make predictions
trainPredict = modelTest.predict(trainX)[testLen-look_back:]
testPredict=nextPredict(modelTest,testX,len(dataset)-look_back)[testLen-look_back:]
futurePredict=nextPredict(model,futureX,100)
trainY=trainY[testLen-look_back:]
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
futurePredict = scaler.inverse_transform(futurePredict)
dataset=scaler.inverse_transform(dataset)
# calculate root mean squared error
rmse(trainY[0,:],trainPredict[:,0],"Train")
rmse(trainY[0,:],testPredict[:,0],"Test")
# shift data for plotting
trainPredictPlot=setPlot(trainPredict,len(dataset),testLen,testLen+len(trainPredict))
testPredictPlot=setPlot(testPredict,len(dataset),testLen,testLen+len(testPredict))
futurePredictPlot=setPlot(futurePredict,len(dataset)+len(futurePredict),len(dataset),len(dataset)+len(futurePredict))
#evaluate
print("input data:")
evaluate(dataset)
print("predict data:")
evaluate(futurePredict)
# plot baseline and predictions
testFig=plt.figure("test")
plt.plot(dataset,color='#0B649F',linestyle='-',marker='.',label='input')
plt.plot(trainPredictPlot,color='green',linestyle='',marker='x',label='single step predict')
plt.plot(testPredictPlot,color='orange',linestyle='',marker='+',label='next step predict')
plt.legend(loc='upper left')
predictFig=plt.figure("predict")
plt.plot(dataset,color='#0B649F',linestyle='-',marker='.',label='input')
plt.plot(futurePredictPlot,color='#ff0066',linestyle='-',marker='.',label='predict')
plt.legend(loc='upper left')
#fittingPredictPlot.show()
#fittingPlot.show()
#plt.show()

#save plot
testFig.savefig(path+os.sep+'test.svg')
predictFig.savefig(path+os.sep+'predict.svg')
fittingPredictFig.savefig(path+os.sep+'fittingPredictPlot.svg')
fittingFig.savefig(path+os.sep+'fittingPlot.svg')
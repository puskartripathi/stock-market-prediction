import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from tensorflow import keras

def search(stockName):
    try:
        Image = open('static/stocks/'+stockName+'/'+stockName+'1.png', 'r')

        print("cache found")
        return 1

    except FileNotFoundError:
        print("file not found")
        return 0


def createDataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0] 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def stockpredict(stockName):
    df=pd.read_csv('dataset/UPDATED_NABIL.csv')
   ## print('i am here')
    path = os.getcwd() 
    os.mkdir(path+"/static/stocks/"+stockName)
    df1=df.reset_index()['Close']
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    
     #splitting dataset
    training_size=int(len(df1)*0.65)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
    
 



    time_step = 60
    X_train, y_train = createDataset(train_data, time_step)
    X_test, ytest = createDataset(test_data, time_step)
  
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    
    
    model=Sequential()
    model=Sequential()
    model.add(LSTM(128,activation='relu',return_sequences=True,input_shape=(60,1)))
    model.add(Dropout(0.40))
    model.add(LSTM(64,return_sequences=True))
    model.add(Dropout(0.20))
    model.add(LSTM(32))
    model.add(Dropout(0.10))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=32,verbose=1)


    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    y_train=scaler.inverse_transform(y_train.reshape(-1,1))
    ytest=scaler.inverse_transform(ytest.reshape(-1,1))
    print(math.sqrt(mean_squared_error(y_train,train_predict)))
    
    ### Test Data RMSE
    print(math.sqrt(mean_squared_error(ytest,test_predict)))
    #plt.plot(y_train)
    #plt.plot(train_predict)
    #plt.show()
    
    plt.plot(y_train, color = 'red', label = 'Real Nabil Stock Price')
    plt.plot(train_predict, color = 'blue', label = 'Predicted Nabil Stock Price ')
    plt.title('Nabil bank Stock Price validation on training data')
    plt.xlabel('Time')
    plt.ylabel('Nabil Stock Price')
    plt.legend()
    plt.savefig('static/stocks/'+stockName+'/'+stockName+'1.png')
    plt.clf()
    plt.close()
    
    plt.plot(ytest, color = 'red', label = 'Real Nabil Stock Price')
    plt.plot(test_predict, color = 'blue', label = 'Predicted Nabil Stock Price ')
    plt.title('Nabil bank Stock Price validation on test data')
    plt.xlabel('Time')
    plt.ylabel('Nabil Stock Price')
    plt.legend()
    plt.savefig('static/stocks/'+stockName+'/'+stockName+'2.png')
    plt.clf()
    plt.close()
    
    #plt.plot(ytest)
    #plt.plot(test_predict)
    #plt.show()
    x_input=df1[len(df1)-60:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    n_steps=60
    i=0
    while(i<1):

            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    

    print(scaler.inverse_transform(lst_output))
    return (scaler.inverse_transform(lst_output))
    
    
    




    
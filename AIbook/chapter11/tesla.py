import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import os
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.layers import Embedding
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from functools import partial


def custom_loss(y_true, y_pred, weights):
    return K.square(K.abs(y_true - y_pred) * weights)

data=pd.read_csv("TSLA.csv")
print(data)

x= []
y = []
for i in range(data.shape[0]-5):
  x.append(data.loc[i:(i+4)]['Close'].values)
  y.append(data.loc[i+5]['Close'])


x = np.array(x)
y = np.array(y)

a=x.max(axis=0)

print(x)

print(x.shape, y.shape)

x = x.reshape(x.shape[0],x.shape[1],1)

numOfAllPoints=x.shape[0]
numOfTestPoints=50
numOfTrainPoints=numOfAllPoints-numOfTestPoints


X_train = x[:numOfTrainPoints,:,:]
y_train = y[:numOfTrainPoints]

X_test = x[numOfTrainPoints:,:,:]
y_test = y[numOfTrainPoints:]

print("y_test.shape",y_test.shape)

input_layer = Input(shape=(5,1))
weights_tensor = Input(shape=(1,))

i1 = Dense(100, activation='relu')(input_layer)
i2 = LSTM(100)(i1)
i3 = Dense(1000, activation='relu')(i2)
out = Dense(1, activation='linear')(i3)
model = Model([input_layer, weights_tensor], out)
model.summary()

cl = partial(custom_loss, weights=weights_tensor)

weights = np.arange(X_train.shape[0]).reshape((X_train.shape[0]),1)/numOfTrainPoints


adam = Adam(lr=0.0001)
test_weights = np.ones((numOfTestPoints,1))



model = Model([input_layer, weights_tensor], out)
model.compile(adam, cl)
model.fit(x=[X_train, weights], y=y_train, epochs=100,batch_size = 32, validation_data = ([X_test, test_weights], y_test))



#save network
# serialize model to JSON
model_json = model.to_json()
with open("tesla_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("tesla_model.h5")
print("Saved model to disk")


pred = model.predict([X_test,test_weights])
print(pred)

plt.figure(figsize=(6,6))
plt.plot(y_test,'r',label='actual')
plt.plot(pred,'--', label = 'predicted')

plt.title('Variation of actual and predicted stock price')
plt.ylabel('Stock price')
plt.legend()

plt.show()



import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist

from keras import optimizers
from matplotlib import  pyplot 



(x_train,y_train),(x_test,y_test)=mnist.load_data()

#define parameters
input_number=28*28
hidden_number=50
num_of_categories=10
minibatch_size=10
num_of_epochs=3
sgd = optimizers.SGD(lr=3)

#prepare labels
print("y_train.shape: ",y_train.shape)
y_train=to_categorical(y_train)
print("categoral y_train.shape: ",y_train.shape)
y_test=to_categorical(y_test)
print("y_test.shape=",y_test.shape)

#reshape x_train
print("x_train.shape:",x_train.shape,"  x_train.dtype: ",x_train.dtype)
x_train=np.reshape(x_train,[x_train.shape[0],input_number])
x_train=x_train.astype('float32')
x_train=x_train/255
x_test=np.reshape(x_test,[x_test.shape[0],input_number])
x_test=x_test.astype('float32')
#normalize x_train
x_test=x_test/255
print("reshaped x_train.shape:",x_train.shape,"  x_train.dtype: ",x_train.dtype)

model=Sequential()
model.add(Dense(hidden_number,input_dim=input_number))
model.add(Activation('sigmoid'))
model.add(Dense(num_of_categories))
model.add(Activation('sigmoid'))
model.summary()
plot_model(model,to_file="first_model.png",show_shapes=True)
model.compile(loss='mean_squared_error', optimizer=sgd,metrics=['accuracy'])

print(x_train.shape)

#train the network
history=model.fit(x_train,y_train,validation_split=0.1,epochs=num_of_epochs,batch_size=minibatch_size)

#evaluate against test data
loss,acc=model.evaluate(x_test,y_test,batch_size=minibatch_size)
print('loss=',loss,'acc=',acc)

#plot train and test accurecy and loss
pyplot.figure()
pyplot.subplot(211)
pyplot.plot(history.epoch,history.history['acc'])
pyplot.ylabel('train accurancy')
pyplot.grid(True)

pyplot.subplot(212)
pyplot.plot(history.epoch,history.history['val_acc'])
pyplot.xlabel('epoch number')
pyplot.ylabel('test accurancy')
pyplot.grid(True)
pyplot.show()

pyplot.figure()
pyplot.subplot(211)
pyplot.plot(history.epoch,history.history['loss'])
pyplot.ylabel('train loss')
pyplot.grid(True)

pyplot.subplot(212)
pyplot.plot(history.epoch,history.history['val_loss'])
pyplot.xlabel('epoch number')
pyplot.ylabel('test loss')
pyplot.grid(True)
pyplot.show()







import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist
from keras import optimizers
from keras.regularizers import l2,l1
from matplotlib import  pyplot ,image
import cv2


(x_train,y_train),(x_test,y_test)=mnist.load_data()



#define parameters
input_number=28*28
hidden_number=50
num_of_categories=10
minibatch_size=10
num_of_epochs=30
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
x_train_1000=x_train[:1000]
y_train_1000=y_train[:1000]
history=model.fit(x_train_1000,y_train_1000,validation_split=0.1,epochs=num_of_epochs,batch_size=minibatch_size)

pyplot.figure(figsize=(7,5))
pyplot.subplot(211)

pyplot.plot(history.epoch,history.history['acc'],label='train')
pyplot.plot(history.epoch,history.history['val_acc'], linestyle='dashed',label='test')
pyplot.ylabel('accurancy')
pyplot.xlabel('epoch number')
pyplot.grid(True)
pyplot.legend()
pyplot.subplot(212)
pyplot.plot(history.epoch,history.history['loss'],label='train')
pyplot.plot(history.epoch,history.history['val_loss'], linestyle='dashed',label='test')
pyplot.ylabel('loss')
pyplot.xlabel('epoch number')
pyplot.grid(True)
pyplot.legend()
pyplot.subplots_adjust(hspace=0.5)
pyplot.show()



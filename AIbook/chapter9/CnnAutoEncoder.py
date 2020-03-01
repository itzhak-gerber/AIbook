import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Conv2DTranspose,Reshape
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
num_of_epochs=5

adam = optimizers.Adam()

#prepare labels
print("y_train.shape: ",y_train.shape)
y_train=to_categorical(y_train)
print("categoral y_train.shape: ",y_train.shape)
y_test=to_categorical(y_test)
print("y_test.shape=",y_test.shape)

#reshape x_train
print("x_train.shape:",x_train.shape,"  x_train.dtype: ",x_train.dtype)
x_train=np.reshape(x_train,(x_train.shape[0],28,28,1))
x_train=x_train.astype('float32')
x_train=x_train/255
x_test=np.reshape(x_test,(x_test.shape[0], 28,28,1))
x_test=x_test.astype('float32')
#normalize x_train
x_test=x_test/255
print("reshaped x_train.shape:",x_train.shape,"  x_train.dtype: ",x_train.dtype)

model=Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten(name="vector"))
model.add(Reshape((1,1,32)))
model.add(Conv2DTranspose(8,kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(Conv2DTranspose(16,kernel_size=(5,5)))
model.add(Activation('relu'))
model.add(Conv2DTranspose(32,kernel_size=(8,8)))
model.add(Activation('relu'))
model.add(Conv2DTranspose(32,kernel_size=(15,15)))
model.add(Activation('relu'))
model.add(Conv2DTranspose(1,kernel_size=(3,3),padding='same'))
model.add(Activation('relu'))
model.summary()

model.compile(loss='mean_squared_error',
              optimizer='adam')
            

model.fit(x_train, x_train,
              batch_size=1024,
              epochs=100,
              validation_data=(x_test, x_test),
              verbose=1)

#save network
# serialize model to JSON
model_json = model.to_json()
with open("vector_cnn_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("vector_cnn_model.h5")
print("Saved model to disk")






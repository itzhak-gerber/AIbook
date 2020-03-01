import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Conv2DTranspose
from keras.layers import Conv2D, MaxPooling2D,Reshape
import os
from matplotlib import  pyplot ,image
import numpy as np




def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def ShowImages(images,fileName):
    pyplot.figure(figsize=(10,10))
    num=0
    for i in range(3):
        for j in range(3):
            k=i*4+j+1
            num+=1
            sub=pyplot.subplot(4,4,k)
            pyplot.axis('off')
            sub.set_title("image "+str(num))
            pyplot.imshow(images[num])
    pyplot.savefig(fileName)
    pyplot.show()
   


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# convert color train and test images to gray
x_train_gray = rgb2gray(x_train)
x_test_gray = rgb2gray(x_test)

ShowImages(x_test,"colorImages.png")
ShowImages(x_test_gray,"grayImages.png")


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255




x_train_gray = x_train_gray.astype('float32')
x_test_gray = x_test_gray.astype('float32')
x_train_gray /= 255
x_test_gray /= 255

x_train_gray=np.reshape(x_train_gray,(x_train_gray.shape[0],32,32,1))
x_test_gray=np.reshape(x_test_gray,(x_test_gray.shape[0],32,32,1))

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',strides=2, input_shape=(32,32,1)))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',strides=2))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same',strides=2))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(4096))
model.add(Reshape((4,4,256)))
model.add(Conv2DTranspose(256,strides=2,kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2DTranspose(128,strides=2,kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2DTranspose(64,strides=2,kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2DTranspose(3,kernel_size=(3,3), padding='same'))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer='adam')
            

model.fit(x_train_gray, x_train,
              batch_size=32,
              epochs=100,
              validation_data=(x_test_gray, x_test),
              verbose=1)


#save network
# serialize model to JSON
model_json = model.to_json()
with open("colorization_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("colorization_model.h5")
print("Saved model to disk")



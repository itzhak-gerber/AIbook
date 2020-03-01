import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist
from keras import optimizers
from matplotlib import  pyplot ,image
from keras.models import model_from_json
from keras.datasets import cifar10
import cv2;



# load json and create model
json_file = open('colorization_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("colorization_model.h5")
print("Loaded model from disk")



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

x_train_gray = x_train_gray.astype('float32')
x_test_gray = x_test_gray.astype('float32')
x_train_gray /= 255
x_test_gray /= 255

x_train_gray=np.reshape(x_train_gray,(x_train_gray.shape[0],32,32,1))
x_test_gray=np.reshape(x_test_gray,(x_test_gray.shape[0],32,32,1))


result = model.predict(x_test_gray[0:10])

ShowImages(result,"ResultImages.png")


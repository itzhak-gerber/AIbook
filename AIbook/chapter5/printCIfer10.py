import keras
from keras.datasets import cifar10
from matplotlib import  pyplot ,image
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

classes={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}


pyplot.figure()
for i in range(1,10):
    sub=pyplot.subplot(3,3,i)
    pyplot.axis('off')
    sub.set_title(classes[y_train[i][0]])
    pyplot.imshow(x_train[i])
    
   

pyplot.show()



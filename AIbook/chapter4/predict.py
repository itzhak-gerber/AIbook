import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist
from keras import optimizers
from matplotlib import  pyplot ,image
from keras.models import model_from_json
import cv2;


x1 = cv2.imread('image0.png')
x1 = cv2.cvtColor(x1, cv2.COLOR_BGR2GRAY )
pyplot.imshow(x1)
pyplot.show()
x1=x1.astype('float32')/255
x1=np.reshape(x1,(1,28*28))

lx=[]
for i in range(5):
    x=cv2.imread('image'+str(i)+'.png')
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY )
    x=x.astype('float32')/255
    x=np.reshape(x,x.shape[0]*x.shape[1])
    lx.append(x)


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#predict One image
result=loaded_model.predict(x1)
print("result=",result[0])
l=np.argmax(result[0])
print("l=",l)

#predict several images
mat= np.array(lx)
resultMat=loaded_model.predict(mat)
l=np.argmax(resultMat,1)
print(l)






import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras import optimizers
from matplotlib import  pyplot ,image
from keras.models import model_from_json
import cv2;
from keras.applications.vgg16 import preprocess_input


x1 = cv2.imread("./test/fiat.jpg")
x1=cv2.resize(x1,(300,300))
x1 = cv2.cvtColor(x1, cv2.COLOR_BGR2RGB )
pyplot.imshow(x1)
pyplot.show()

img=preprocess_input(x1.reshape(1,300,300,3))


x2 = cv2.imread("./test/volvo.jpg")
x2=cv2.resize(x2,(300,300))
x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2RGB )
pyplot.imshow(x2)
pyplot.show()

img2=preprocess_input(x2.reshape(1,300,300,3))


# load json and create model
json_file = open('modelCars.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelCars.h5")
print("Loaded model from disk")


def getClass(x):
    if abs(x-1)<abs(x-0):
        return "volvo"
    else:
        return "fiat"

#predict One image
result=loaded_model.predict(img)
print("imag=",getClass(result[0]))



result=loaded_model.predict(img2)
print("imag2=",getClass(result[0]))


from keras import models
feature_extraction_model=models.Model(inputs=loaded_model.input,outputs=loaded_model.layers[1].output)

features=feature_extraction_model.predict(img)
print("features.shape",features.shape)

pyplot.figure()
pyplot.imshow(features[0][:,:,0])
pyplot.show()



pyplot.figure(figsize=(10,10))
for i in range(3):
    for j in range(3):
        k=i*4+j+1
        sub=pyplot.subplot(4,4,k)
        pyplot.axis('off')
        sub.set_title("filter "+str(k))
        pyplot.imshow(features[0][:,:,k])
    
   

pyplot.show()
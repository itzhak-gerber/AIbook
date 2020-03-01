from keras.models import model_from_json
from matplotlib import  pyplot ,image
import cv2;
import numpy as np


classes={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}


# load json and create model
json_file = open('modelFunctionalAPI.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights.best.hdf5")
print("Loaded model from disk")

img = cv2.imread("fiat.jpg")
img=cv2.resize(img,(32,32))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
pyplot.imshow(img)
pyplot.show()

img=img/255
img=img.reshape((1,32,32,3))

res=loaded_model.predict([img,img])
l= np.argmax(res)
print(classes[l])










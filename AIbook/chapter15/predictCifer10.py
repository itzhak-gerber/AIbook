from keras.models import model_from_json
import cv2;
from matplotlib import  pyplot 
import numpy as np


classes={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}

#orgImg = cv2.imread('cat.jpg')
#orgImg = cv2.cvtColor(orgImg, cv2.COLOR_BGR2RGB)

#pyplot.imshow(orgImg)
#pyplot.show()


#img=cv2.resize(orgImg,(32,32))
#pyplot.imshow(img)
#pyplot.show()

## load json and create model
#json_file = open('modelflaskcifer10.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("modelflaskcifer10.h5")
#print("Loaded model from disk")

#img=img/255
#sample=img.reshape(1,32,32,3)

##predict One image
#result=loaded_model.predict(sample)
#print("result=",result[0])
#l=np.argmax(result[0])
#print("l=",l)
#print(classes[l])



def FlaskPredict(orgImg):
    orgImg = cv2.cvtColor(orgImg, cv2.COLOR_BGR2RGB)
    img=cv2.resize(orgImg,(32,32))
    # load json and create model
    json_file = open('modelflaskcifer10.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("modelflaskcifer10.h5")
    print("Loaded model from disk")

    img=img/255
    sample=img.reshape(1,32,32,3)

    #predict One image
    result=loaded_model.predict(sample)
    l=np.argmax(result[0])
    s=classes[l]
    return s
    

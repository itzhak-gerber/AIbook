import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from copy import deepcopy
from keras.models import model_from_json


# load json and create model
json_file = open('segmentation_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("segmentation_model.h5")
print("Loaded model from disk")

dir_images=".\\data\\horses\\images\\"



img = cv.imread(dir_images+"image-69.png")
img = cv.resize(img,(224,224))

img=img.reshape(1,224,224,3)

result = model.predict(img);

ImgResult= result.reshape(224,224)
OrgImag=img.reshape(224,224,3)

maskResult=ImgResult*255

masked_image = np.zeros_like(OrgImag)
masked_image[:,:,0] = maskResult
masked_image[:,:,1] = maskResult
masked_image[:,:,2] = maskResult



imgray = cv.cvtColor(masked_image, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
result = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contourList=[]
for i in range(len(result[1])):
    contourList.append(result[1][i])


maxSize=0
for cont in contourList:
    if cont.shape[0]>maxSize:
        maxSize=cont.shape[0]
        maxContour=cont


lx=[]
ly=[]
for i in range(maxContour.shape[0]):
    x=maxContour[i][0][0]
    y=maxContour[i][0][1]
    lx.append(x)
    ly.append(y)

dir_images=".\\data\\horses\\images\\"
img = cv.imread(dir_images+"image-69.png")
img = cv.resize(img,(224,224))

for i in range(len(lx)):
    cv.circle(img,(lx[i],ly[i]),3,(0,255,255))

plt.imshow(img)
plt.show()

img = cv.imread(dir_images+"image-69.png")
img = cv.resize(img,(224,224))

for i in range(len(lx)-1):
    p1=(lx[i],ly[i])
    p2=(lx[i+1],ly[i+1])
    cv.line(img,p1,p2,(0,255,255))

plt.imshow(img)
plt.show()


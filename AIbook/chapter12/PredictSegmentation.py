from keras.models import model_from_json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2



# load json and create model
json_file = open('segmentation_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("segmentation_model.h5")
print("Loaded model from disk")

dir_images=".\\data\\horses\\images\\"



img = cv2.imread(dir_images+"image-69.png")
img = cv2.resize(img,(224,224))



img=img.reshape(1,224,224,3)

result = model.predict(img);

ImgResult= result.reshape(224,224)
OrgImag=img.reshape(224,224,3)

saveResult=ImgResult*255

cv2.imwrite('maskResult.jpg',saveResult)

plt.figure()
plt.subplot(2,1,1)
plt.imshow(OrgImag)
plt.subplot(2,1,2)
plt.imshow(ImgResult)
plt.show()



from keras.applications.vgg16 import VGG16 
from keras.layers import Input, Conv2D, concatenate, UpSampling2D, BatchNormalization, Activation, Cropping2D, ZeroPadding2D
from keras.layers import Input, merge, Conv2D, MaxPooling2D,UpSampling2D, Dropout, Cropping2D, merge, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.models import Model
import cv2
from scipy import ndimage
import os
import numpy as np
from matplotlib import pyplot
from tensorflow.keras.utils import plot_model


dir_images=".\\data\\horses\\images\\"
image_files=os.listdir(dir_images)
dir_masks=".\\data\\horses\\masks\\"
mask_files=os.listdir(dir_masks)

x = []
y = []
for file_name in image_files:
  img = cv2.imread(dir_images+file_name)
  img = cv2.resize(img,(224,224))
  x.append(img)


for file_name in mask_files:
  img = cv2.imread(dir_masks+file_name)
  img = cv2.resize(img,(224,224))
  y.append(img)
  

x_train = np.array(x)
y = np.array(y)


y = np.where(y>250,1,0)


y = np.array(y)[:,:,:,0]
y_train = y.reshape(y.shape[0],y.shape[1],y.shape[2],1)



print("x.shape",x_train.shape)
print("y.shape",y_train.shape)


vgg16_no_top = VGG16(input_shape =  (224,224,3), include_top = False, weights = 'imagenet')
vgg16_no_top.trainable = False
vgg16_no_top.summary()


left1 = Model(inputs=vgg16_no_top.input,outputs=vgg16_no_top.get_layer('block1_conv2').output).output
left2 = Model(inputs=vgg16_no_top.input,outputs=vgg16_no_top.get_layer('block2_conv2').output).output
left3 = Model(inputs=vgg16_no_top.input,outputs=vgg16_no_top.get_layer('block3_conv3').output).output
left4 = Model(inputs=vgg16_no_top.input,outputs=vgg16_no_top.get_layer('block4_conv3').output).output
left5 = Model(inputs=vgg16_no_top.input,outputs=vgg16_no_top.get_layer('block5_conv3').output).output

right5=UpSampling2D(size =(2,2))(left5)
up4 = Conv2D(512, 2, activation = 'relu', padding = 'same')(right5)
merge4 = concatenate([left4,up4], axis = 3)

conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge4)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)
conv6 = BatchNormalization()(conv6)
right6=UpSampling2D(size =(2,2))(conv6)
up3 = Conv2D(256, 2, activation = 'relu', padding = 'same')(right6)
merge3 = concatenate([left3,up3], axis = 3)

conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge3)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)
conv7 = BatchNormalization()(conv7)
right7=UpSampling2D(size =(2,2))(conv7)
up2 = Conv2D(128, 2, activation = 'relu', padding = 'same')(right7)
merge2 = concatenate([left2,up2],axis = 3)

conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge2)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)
conv8 = BatchNormalization()(conv8)
right8=UpSampling2D(size =(2,2))(conv8)
up1 = Conv2D(64, 2, activation = 'relu', padding = 'same')(right8)

merge1 = concatenate([left1,up1], axis = 3)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge1)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
conv9 = BatchNormalization()(conv9)

conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

model = Model(input = vgg16_no_top.input, output = conv10)
model.summary()

plot_model(model, to_file='segmentation.png', show_shapes=True)

print(model.layers[17].name)

for layer in model.layers[:18]:
  layer.trainable = False
  

model.compile(optimizer='Adam', 
                   loss='binary_crossentropy', metrics = ['accuracy'])
np.max(x)
np.sum(y)
history = model.fit(x_train,y_train,epochs=3,batch_size=1,validation_split=0.1)

#save network
# serialize model to JSON
model_json = model.to_json()
with open("segmentation_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("segmentation_model.h5")
print("Saved model to disk")

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1, len(val_loss_values) + 1)
pyplot.subplot(211)
pyplot.plot(epochs, history.history['loss'], 'ro', label='Training loss')
pyplot.plot(epochs, val_loss_values, 'b', label='Test loss')
pyplot.title('Training and test loss')
pyplot.xlabel('Epochs')
pyplot.ylabel('Loss')
pyplot.legend()
pyplot.grid('off')
pyplot.show()
pyplot.subplot(212)
pyplot.plot(epochs, history.history['acc'], 'ro', label='Training accuracy')
pyplot.plot(epochs, val_acc_values, 'b', label='Test accuracy')
pyplot.title('Training and test accuracy')
pyplot.xlabel('Epochs')
pyplot.ylabel('Accuracy')
pyplot.legend()
pyplot.grid('off')
pyplot.show()
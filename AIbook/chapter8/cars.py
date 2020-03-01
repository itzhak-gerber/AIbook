import os

from matplotlib import pyplot,image
import numpy as np
import cv2
from keras_preprocessing.image import ImageDataGenerator

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.applications import  vgg16
from keras.applications.vgg16 import preprocess_input
from keras.utils.vis_utils import plot_model
from keras.applications.vgg16 import decode_predictions
from keras import models
from keras.models import model_from_json
from keras.models import Model



vgg16_model = vgg16.VGG16(include_top=False, weights='imagenet',input_shape=(300,300,3))

x=vgg16_model.output
x=GlobalAveragePooling2D(name="myGlobalPooling")(x)
x=Dense(512,activation='relu',name="myFirstDense")(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dropout(0.25)(x)
x=Dense(512,activation='relu',name="mySecondDense")(x) #dense layer 2
x=Dropout(0.25)(x)
x=Dense(256,activation='relu',name="myThirdDense")(x) #dense layer 3
x=Dropout(0.25)(x)
preds=Dense(1,activation='sigmoid')(x) #final layer with softmax activation
model=Model(inputs=vgg16_model.input,outputs=preds)


model.summary()

for i,layer in enumerate(model.layers):
  print(i,layer.name)

for layer in model.layers[:19]:
    layer.trainable=False
for layer in model.layers[19:]:
    layer.trainable=True

for i,layer in enumerate(model.layers):
  print(i," "+layer.name,layer.trainable)



train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
validate_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_datagen.flow_from_directory(
    directory=r"./train/",
    target_size=(300, 300),
    color_mode="rgb",
    batch_size=4,
    class_mode="binary",
    shuffle=True,
    seed=42
)




validate_generator = validate_datagen.flow_from_directory(
    directory=r"./validate/",
    target_size=(300, 300),
    color_mode="rgb",
    batch_size=4,
    class_mode="binary",
    shuffle=True,
    seed=42
)


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALIDATE=validate_generator.n//validate_generator.batch_size
print("STEP_SIZE_TRAIN",STEP_SIZE_TRAIN)
print("STEP_SIZE_VALIDATE",STEP_SIZE_VALIDATE)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validate_generator,
                    validation_steps=STEP_SIZE_VALIDATE,
                    epochs=10
                    )

#save network
# serialize model to JSON
model_json = model.to_json()
with open("modelCars.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelCars.h5")
print("Saved model to disk")
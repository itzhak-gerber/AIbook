import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,concatenate
from keras.layers import Conv2D, MaxPooling2D, Input
import os
from keras import Model
from tensorflow.keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

batch_size = 32
num_classes = 10
epochs = 10
num_filters=32
kernel_size=(3,3)
dropout=0.25

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
print(y_train[0:3])
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train[0:3])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

shape=(32,32,3)
input1 = Input(shape=shape)
x1 = input1
filters = num_filters
# 3 layers of Conv2D-Dropout-MaxPooling2D
# number of filters doubles after each layer (32-64-128)
for i in range(3):
    x1 = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu')(x1)
    x1 = Dropout(dropout)(x1)
    x1 = MaxPooling2D()(x1)
    filters *= 2


input2 = Input(shape=shape)
x2 = input2
filters = num_filters
# 3 layers of Conv2D-Dropout-MaxPooling2D
# number of filters doubles after each layer (32-64-128)
for i in range(3):
    x2 = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu')(x2)
    x2 = Dropout(dropout)(x2)
    x2 = MaxPooling2D()(x2)
    filters *= 2

x = concatenate([x1, x2])

print("x.shape",x.shape)

x = Flatten()(x)
x = Dropout(dropout)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(dropout)(x)
outputs = Dense(num_classes, activation='softmax')(x)






model = Model([input1,input2], outputs)

plot_model(model, to_file='functionalAPI.png', show_shapes=True)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])

model.summary()

#save network
# serialize model to JSON
model_json = model.to_json()
with open("modelFunctionalAPI.json", "w") as json_file:
    json_file.write(model_json)

# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit([x_train,x_train], y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=([x_test,x_test], y_test),
              shuffle=True,callbacks=callbacks_list, verbose=1)

# Score trained model.
scores = model.evaluate([x_test,x_test], y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])




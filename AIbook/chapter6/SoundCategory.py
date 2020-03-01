import librosa
import librosa.display
from matplotlib import  pyplot ,image
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split

num_cores = 4

CPU=True
GPU=False

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU}
                       )

session = tf.Session(config=config)
K.set_session(session)


relative="content\\genres\\"

filename = 'blues.00000.wav'

dirs=[d  for d in os.listdir(relative) if os.path.isdir(relative+d)]
num_classes=len(dirs)

dic={}
i=0
for dir in dirs:
    dic[dir]=i
    i+=1



melSpectograms=[]
genres=[]

for dir in dirs:
    print("dir=",dir)

    fileNames=os.listdir(relative+dir)
   
    for f in fileNames:
       filePath=relative+dir+"\\"+f
       y, sr = librosa.load(filePath)
       sr=16000
       mel = librosa.feature.melspectrogram(y, sr=sr,n_fft = 2048)
       mel=mel[:128,:1280]
       melSpectograms.append(mel.T)
       genres.append(dic[dir])

print("len(melSpectograms)=",len(melSpectograms))
print(melSpectograms[0].shape)
x_train=np.array(melSpectograms)
genres = keras.utils.to_categorical(genres, num_classes)

print("x_train")
print(x_train.shape)
print("genres")
print(genres.shape)
input_shape = (x_train.shape[1], x_train.shape[2])


model = Sequential()
model.add(Conv1D(64,  3,  activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2, strides=2))

model.add(Conv1D(128,  3,  activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2, strides=2))

model.add(Conv1D(256,  3,  activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2, strides=2))

model.add(Conv1D(512,  3,  activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2, strides=2))

model.add(Conv1D(1024,  3,  activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2, strides=2))

model.add(Conv1D(2048,  3,  activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2, strides=2))

model.add(Conv1D(4096,  3,  activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2, strides=2))

model.add(GlobalMaxPooling1D())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

adam = keras.optimizers.Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

model.summary()


x_train, x_test, y_train, y_test = train_test_split(x_train,genres,test_size=0.1)


history = model.fit(x_train, y_train,
      batch_size=128,
      epochs=100,
      verbose=1,
      validation_data=(x_test, y_test))           

pyplot.figure()
pyplot.plot(history.epoch,history.history['loss'],label='train')
pyplot.plot(history.epoch,history.history['val_loss'], linestyle='dashed',label='test')
pyplot.show()
pyplot.figure()
pyplot.plot(history.epoch,history.history['acc'],label='train')
pyplot.plot(history.epoch,history.history['val_acc'], linestyle='dashed',label='test')
pyplot.show()

                                 
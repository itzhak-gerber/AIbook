from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from matplotlib import  pyplot ,image

from keras.datasets import boston_housing
import numpy as np
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
print("sample 5 rows")
print(train_data[0:5])
print("labels")
print(train_labels[0:5])

max_labels_range = np.max(train_labels)
max_input_range=np.max(train_data,axis=0)
print("max_labels_range")
print(max_labels_range)
print("max_input_range")
print(max_input_range)
train_data2 = train_data/max_input_range
test_data2 = test_data/max_input_range
train_labels = train_labels/max_labels_range
test_labels = test_labels/max_labels_range

model = Sequential()
model.add(Dense(128, input_dim=13, activation='relu' ))
model.add(Dropout(0.75))
model.add(Dense(32,  activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(1, activation='relu'))
model.summary()

model.compile(loss='mean_absolute_error', optimizer='adam')


history = model.fit(train_data2, train_labels, validation_data=(test_data2, test_labels), epochs=150, batch_size=32, verbose=1)
predicted=model.predict(test_data2)
result=np.mean(np.abs( predicted- test_labels))*max_labels_range
print("result=",str(result))

pyplot.figure(figsize=(7,5))

pyplot.plot(history.epoch,history.history['loss'],label='train')
pyplot.plot(history.epoch,history.history['val_loss'], linestyle='dashed',label='test')
pyplot.ylabel('loss')
pyplot.xlabel('epoch number')
pyplot.show()

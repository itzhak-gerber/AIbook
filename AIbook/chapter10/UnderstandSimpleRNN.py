import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import SimpleRNN

#All you need is love
word_to_int={'All':4,'you':2,'need':5,'is':1,'love':3}
input_words=[['All','you'],['you','need'],['need','is']]
label_word=[['need'],['is'],['love']]
encoded_word=[[4,2],[2,5],[5,1]]
encoded_labels=[[5],[1],[3]]
one_hot_encoded_labels= to_categorical(encoded_labels)
print(one_hot_encoded_labels)

#(batch_size,time_steps,features)
#(3,2,1)
input_words=np.reshape(encoded_word,(3,2,1))
print(input_words.shape)

model=Sequential()
model.add(SimpleRNN(1,activation='tanh',return_sequences=False,recurrent_initializer='Zeros',input_shape=(2,1)))
model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
model.summary()




model.fit(input_words,one_hot_encoded_labels,epochs=500)



#save network
# serialize model to JSON
model_json = model.to_json()
with open("UsimpleRNN_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("UsimpleRNN_model.h5")
print("Saved model to disk")





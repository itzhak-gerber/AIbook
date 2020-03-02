import os
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Embedding,  Dense,SimpleRNN
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import re
import nltk
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from matplotlib import  pyplot




data=pd.read_csv("sentiment.csv")
labels=data['airline_sentiment']

lines=data['text'].values
for s in lines[0:5]:
    print(s)

print(labels)
print(labels[0:5])


stop_words=nltk.corpus.stopwords.words('english')


def PrepareSentence(text):
    text=text.lower()
    text=re.sub('[^0-9a-zA-z]+',' ',text)
    words=text.split()
    results=[]
    for s in words[1:]:
        if (s not in stop_words):
            results.append(s)
    return ' '.join(results)


clean_lines=[]
for line in lines:
    clean_line=PrepareSentence(line)
    clean_lines.append(clean_line)


corpus=[]
for line in clean_lines:
    words=line.split()
    for word in words:
        corpus.append(word)

dic_words={}
for word in corpus:
    if word in dic_words:
        dic_words[word]+=1
    else:
        dic_words[word]=1

words= sorted(dic_words,key=dic_words.get ,reverse=True)      

print(words[0:5])

words_to_int={words[i]:i  for i in range(len(words))}
int_to_words={i:words[i]  for i in range(len(words))}

max_sentence_len=0
numeric_lines=[]
for line in clean_lines:
    words=line.split()
    l=[words_to_int[word] for word in words]
    numeric_lines.append(l)
    length=len(l)
    if length>max_sentence_len:
        max_sentence_len=length

padded_numeric_lines=pad_sequences(numeric_lines, maxlen=max_sentence_len,padding='post',value=0)


x_train,x_test,Y_train,Y_test= train_test_split(padded_numeric_lines,labels,test_size=0.2)

y_train=to_categorical(Y_train)
y_test=to_categorical(Y_test)

embeding_length=32
corpus_len=len(words_to_int)
model=Sequential()
model.add(Embedding(input_dim=corpus_len,output_dim=32,input_length=max_sentence_len))
model.add(SimpleRNN(30,return_sequences=False))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=32)


pyplot.figure(figsize=(7,5))
pyplot.subplot(211)

pyplot.plot(history.epoch,history.history['acc'],label='train')
pyplot.plot(history.epoch,history.history['val_acc'], linestyle='dashed',label='test')
pyplot.ylabel('accurancy')
pyplot.xlabel('epoch number')
pyplot.grid(True)
pyplot.legend()
pyplot.subplot(212)
pyplot.plot(history.epoch,history.history['loss'],label='train')
pyplot.plot(history.epoch,history.history['val_loss'], linestyle='dashed',label='test')
pyplot.ylabel('loss')
pyplot.xlabel('epoch number')
pyplot.grid(True)
pyplot.legend()
pyplot.subplots_adjust(hspace=0.5)
pyplot.show()
       

    






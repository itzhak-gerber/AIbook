from keras.models import model_from_json
import numpy as np

encoded_word=[[4,2],[2,5],[5,1]]
input_words=np.reshape(encoded_word,(3,2,1))

# load json and create model
json_file = open('UsimpleRNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("UsimpleRNN_model.h5")
print("Loaded model from disk")

test=np.reshape(input_words[0],(1,2,1))

result=model.predict(test)

print(result)



wx=model.get_weights()[0]
wh=model.get_weights()[1]
b=model.get_weights()[2]
wd=model.get_weights()[3]
bd=model.get_weights()[4]

print('wx=',wx,' h=',wh,' b=',b)
print('wd')
print(wd)
print('bd')
print(bd)

h1=np.tanh(4*wx+b)
o=np.tanh(2*wx+wh*h1+b)
print('o=',o)

dense_out=o*wd+bd

softmax=np.exp(dense_out)/np.sum(np.exp(dense_out))
print('sofmax')
print(softmax)







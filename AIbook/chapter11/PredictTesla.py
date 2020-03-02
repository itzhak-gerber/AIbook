from keras.models import model_from_json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data=pd.read_csv("TSLA.csv")
print(data)

x= []
y = []

for i in range(data.shape[0]-5):
  x.append(data.loc[i:(i+4)]['Close'].values)
  y.append(data.loc[i+5]['Close'])



x = np.array(x)
y = np.array(y)



print(x.shape, y.shape)

x = x.reshape(x.shape[0],x.shape[1],1)

numOfAllPoints=x.shape[0]
numOfTestPoints=50
numOfTrainPoints=numOfAllPoints-numOfTestPoints

X_train = x[:numOfTrainPoints,:,:]
y_train = y[:numOfTrainPoints]

X_test = x[numOfTrainPoints:,:,:]
y_test = y[numOfTrainPoints:]

# load json and create model
json_file = open('tesla_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("tesla_model.h5")
print("Loaded model from disk")

test_weights = np.ones((numOfTestPoints,1))



l=[X_test, test_weights]
pred = model.predict(X_test)
print("y_test.shape",y_test.shape)
print("pred.shape",pred.shape)



plt.figure(figsize=(6,6))
plt.plot(y_test,'r',label='actual')
plt.plot(pred,'--', label = 'predicted')

plt.title('Variation of actual and predicted stock price')
plt.ylabel('Stock price')
plt.legend()

plt.show()

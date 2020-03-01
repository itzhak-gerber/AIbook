import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist
from keras import optimizers
from matplotlib import  pyplot ,image
from keras.models import model_from_json
import cv2;



(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=np.reshape(x_train,(x_train.shape[0],28,28,1))
x_train=x_train.astype('float32')
x_train=x_train/255

x_test=np.reshape(x_test,(x_test.shape[0],28,28,1))
x_test=x_test.astype('float32')
x_test=x_test/255


# load json and create model
json_file = open('vector_cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("vector_cnn_model.h5")
print("Loaded model from disk")


from keras.models import Model
vector_layer=Model(inputs=model.input,outputs=model.get_layer('vector').output)

vector_layer.summary()

max_points=1000

from sklearn.manifold import TSNE
import pandas as pd 
vector_layer_output=vector_layer.predict(x_test[0:max_points,:])
tsne_model=TSNE(n_components=2,verbose=1,random_state=0)
TsneDigit=tsne_model.fit_transform(vector_layer_output)
tsne_df=pd.DataFrame(TsneDigit,columns=['x','y'])
tsne_df['Digit']=y_test[0:max_points]

print(tsne_df)


cmap = pyplot.cm.get_cmap('jet')
pyplot.scatter(tsne_df['x'], tsne_df['y'], alpha=0.5, c=y_test[0:max_points], cmap=cmap, s=10)
           
pyplot.colorbar()
pyplot.show()





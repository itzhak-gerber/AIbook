import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist
from keras import optimizers
from keras.regularizers import l2,l1
from keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

import tensorflow as tf
from keras import backend as K

#num_cores = 4
#GPU=False
#CPU=True
#if GPU:
#    num_GPU = 1
#    num_CPU = 1
#if CPU:
#    num_CPU = 1
#    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

(x_train,y_train),(x_test,y_test)=mnist.load_data()



#define parameters
input_number=28*28
hidden_number=50
num_of_categories=10
minibatch_size=10
num_of_epochs=30



#prepare labels
print("y_train.shape: ",y_train.shape)
y_train=to_categorical(y_train)
print("categoral y_train.shape: ",y_train.shape)
y_test=to_categorical(y_test)
print("y_test.shape=",y_test.shape)

#reshape x_train
print("x_train.shape:",x_train.shape,"  x_train.dtype: ",x_train.dtype)
x_train=np.reshape(x_train,[x_train.shape[0],input_number])
x_train=x_train.astype('float32')
x_train=x_train/255
x_test=np.reshape(x_test,[x_test.shape[0],input_number])
x_test=x_test.astype('float32')
#normalize x_train
x_test=x_test/255
print("reshaped x_train.shape:",x_train.shape,"  x_train.dtype: ",x_train.dtype)
def CreateModel(LearningRate=0.1,func='sigmoid'):
    sgd = optimizers.SGD(lr=LearningRate)
    model=Sequential()
    model.add(Dense(hidden_number,input_dim=input_number))
    model.add(Activation(func))
    model.add(Dense(num_of_categories))
    model.add(Activation(func))
    #model.summary()
    model.compile(loss='mean_squared_error', optimizer=sgd,metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=CreateModel,epochs=10, verbose=0)
Activations=['relu','sigmoid','tanh']
batch_size = [10, 11,12]
LearningRate=[0.03,0.3,3]


param_grid = dict( LearningRate=LearningRate,batch_size=batch_size,func=Activations)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)

x_train_1000=x_train[:1000]
y_train_1000=y_train[:1000]

grid_result = grid.fit(x_train_1000, y_train_1000)
print(x_train.shape)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



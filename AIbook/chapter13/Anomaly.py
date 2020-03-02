import numpy as np
import pandas as pd
from pyod.utils.data import generate_data
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense,Dropout
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



contamination = 0.1  # percentage of outliers
n_train = 500  # number of training points
n_test = 500  # number of testing points
n_features = 25 # Number of features

X_train, y_train, X_test, y_test = generate_data(
    n_train=n_train, n_test=n_test,
    n_features= n_features, 
    contamination=contamination,random_state=1234)

X_train, y_train=shuffle(X_train,y_train)

num_of_labeled_outlier=np.sum(y_train)
print("num_of_labeled_outlier=",num_of_labeled_outlier)
num_of_test_labeled_outlier=np.sum(y_test)
print("num_of_test_labeled_outlier=",num_of_test_labeled_outlier)

print("y_train")
print(y_train)

pca = PCA(2)
x_pca = pca.fit_transform(X_train)
x_pca = pd.DataFrame(x_pca)
x_pca.columns=['PC1','PC2']

lx=x_pca['PC1']
ly=x_pca['PC2']

print("after pca")
print(x_pca)


plt.scatter(lx,ly,c=y_train, alpha=0.8)
plt.title('Scatter plot of pca')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

input_dim=25
input_layer = Input(shape=(input_dim, ))
encoder = Dense(25, activation="relu")(input_layer)
encoder = Dropout(0.2)(input_layer)
encoder = Dense(25, activation="relu")(encoder)
encoder = Dropout(0.2)(encoder)
encoder = Dense(25, activation="relu")(encoder)
encoder = Dropout(0.2)(encoder)


encoder = Dense(2, activation="relu")(encoder)
encoder = Dropout(0.2)(encoder)
decoder = Dense(2, activation='relu')(encoder)
decoder = Dropout(0.2)(decoder)

decoder = Dense(25, activation='relu')(encoder)
decoder = Dense(25, activation='relu')(decoder)
decoder = Dense(input_dim)(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()

nb_epoch = 100
batch_size = 50
autoencoder.compile(optimizer='adam', loss='mse' )

history = autoencoder.fit(X_train, X_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.1,
                        verbose=1
                        ).history

plt.figure(figsize=(12,8))
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.ylabel('Loss',fontsize= 18)
plt.xlabel('Epoch',fontsize= 18)
plt.legend(['train', 'test'], loc='upper right', fontsize= 18);
plt.show()


predictions=autoencoder.predict(X_train)
mse = np.mean(np.power(X_train - predictions, 2), axis=1)
print(mse)
outlier = np.where(mse<2,0,1)

print("outlier")
print(outlier)

count_outlier=np.sum(outlier)
print("Num of found outlier",count_outlier)

plt.hist(mse, bins='auto')  
plt.title("Histogram for Model  Anomaly Scores")
plt.show()


diff_outlier=outlier-y_train
print(diff_outlier)
count=np.sum(np.abs(diff_outlier))
print("count=",count)



predictions=autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
print(mse)
outlier = np.where(mse<2,0,1)


print("test outlier")
print(outlier)

count_outlier=np.sum(outlier)
print("Num of  test outlier ",count_outlier)

plt.hist(mse, bins='auto')  
plt.title("Histogram for test Model  Anomaly Scores")
plt.show()


diff_outlier=outlier-y_test
print(diff_outlier)
count=np.sum(np.abs(diff_outlier))
print("test count=",count)





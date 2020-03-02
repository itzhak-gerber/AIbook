import numpy as np
from keras.datasets import mnist
from matplotlib import  pyplot as plt
from sklearn.cluster import KMeans
(x_train,y_train),(x_test,y_test)=mnist.load_data()

input_number=28*28
num_of_samples=1000
x_train=np.reshape(x_train,[x_train.shape[0],input_number])
kmeans = KMeans(n_clusters=10)
part=x_train[0:num_of_samples,:]
kmeans_10 = kmeans.fit_predict(part)

print(kmeans.cluster_centers_.shape)

res=np.zeros((10,10))#(center,digit)
for i in range(num_of_samples):
    center=kmeans_10[i]
    digit=y_train[i]
    res[center][digit]+=1

print(res)

l=[]
r=[]
for i in range(10):
    row=res[i]
    maxindex=np.argmax(row)
    l.append(maxindex)
    m=np.max(row)
    s=np.sum(row)
    r.append(m/s)

print(l)
print(r)


y_pos = np.arange(10)


plt.bar(y_pos, r, align='center', alpha=0.5)
plt.xticks(y_pos, l)
plt.ylabel('probability')
plt.title('digits probability')

plt.show()







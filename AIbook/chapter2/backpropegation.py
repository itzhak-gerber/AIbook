import numpy as np
import matplotlib.pyplot as plt
import keras.datasets.mnist as mnist



#Load data
((trainData,trainLabels),(testData,testLabels))= mnist.load_data();

print("trainData.shape: ",trainData.shape)
print("tranLables.shape",trainLabels.shape)

print("testData.shape",testData.shape)

print("testLabels.shape",testLabels.shape)

print(trainData[0])
print()
      

def ToOneHot(n):
    return [ (int)(i==n) for i in range(10)]

#prepare data   
firstDigit = np.where(trainData[0]>0, 1, 0)
print(firstDigit)
print(trainLabels[0])

trainData=np.where(trainData>0, 1, 0)
testData=np.where(testData>0, 1, 0)

trainData=trainData.reshape(trainData.shape[0],-1)
testData=testData.reshape(testData.shape[0],-1)

print("trainData.shape after reshape: ",trainData.shape)

print("testData.shape after reshape: ",testData.shape)


oneHotTrainLebels=np.ndarray((trainLabels.shape[0],10))
for i in range(len(trainLabels)):
    oneHotTrainLebels[i]=ToOneHot(trainLabels[i])

print("oneHotTrainLebels.shape: ",oneHotTrainLebels.shape)


oneHotTestLebels=np.ndarray((testLabels.shape[0],10))
for i in range(len(testLabels)):
    oneHotTestLebels[i]=ToOneHot(testLabels[i])

print("oneHotTestLebels.shape: ",oneHotTestLebels.shape)


trainDataTuples=[]
for i in range(len(trainData)):
    trainDataTuples.append((trainData[i],oneHotTrainLebels[i]))

print("trainDataTuples[0]: ",type(trainDataTuples[0]))
print("trainDataTuples[0].shape: (",trainDataTuples[0][0].shape,",",trainDataTuples[0][1].shape,")")


testDataTuples=[]
for i in range(len(testData)):
    testDataTuples.append((testData[i],oneHotTestLebels[i]))


#Create network
w1=np.random.randn(50,784)
w2=np.random.randn(10,50)
print("w1.shape: ",w1.shape)
print("w2.shape:", w2.shape )

b1=np.random.randn(50,1)
b2=np.random.randn(10,1)
print("b1.shape: ",b1.shape)
print("b2.shape:", b2.shape )


       

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


#feed foreword digit 0

x=trainData[0].reshape(trainData[0].shape[0],1)
print("Transpose first digit")
print(x.shape)
u1=np.dot(w1,x)
print(u1.shape)
z1=u1+b1
f1=sigmoid(z1)
u2=np.dot(w2,f1)
z2=u2+b2
f2=sigmoid(z2)
print("f1")
print(f1)
print()
print("f2")
print(f2)



def LosFunctionGradiant(a,y):
    return a-y

def sigmoidDerivative(x):
    return sigmoid(x)*(1-sigmoid(x))



def CalcError(y,a):
    e=pow(y-a,2)
    res=(1/2)* e
    s=np.sum(res,0)
    return s


epochs=50
minibatchSize=10
eta=3


for i in range(epochs):
     numOfTrue=0
     numOfFalse=0
     print("epoch "+str(i))
     globalError=0
     np.random.shuffle(trainDataTuples)
     numOfMinibatchs=(int)(len(trainDataTuples)/minibatchSize)
     minibatchPos=0  
    
     for j in range(numOfMinibatchs):
         minibatchPos=j*minibatchSize
         minibatchData=trainDataTuples[minibatchPos:minibatchPos+minibatchSize]
         w2GradientErr=np.zeros(w2.shape)
         w1GradientErr=np.zeros(w1.shape)
         b1GradientErr=np.zeros(b1.shape)
         b2GradientErr=np.zeros(b2.shape)
         eMiniBatch=0
         for i in range(minibatchSize):
            miniTuple=minibatchData[i]
            x=miniTuple[0].reshape(miniTuple[0].shape[0],1)
            y=miniTuple[1].reshape(miniTuple[1].shape[0],1)
           

            #feedforword
            a0=x
            z1=np.dot(w1,x)+b1
            a1=sigmoid(z1)
            z2=np.dot(w2,a1)+b2
            a2=sigmoid(z2)

            c=np.argmax(a2)
            l=np.argmax(y)
            if (c==l):
                numOfTrue+=1
            else:
                numOfFalse+=1

            #errL  calculate last level error
            errL=LosFunctionGradiant(a2,y)*sigmoidDerivative(z2)
            v2=np.dot(errL,a1.T)
            w2GradientErr+=v2
            b2GradientErr+=errL
           

            #calculate last-1 level error
            errF=np.dot((w2.T),errL)*sigmoidDerivative(z1)
            v1=np.dot(errF,a0.T)
            w1GradientErr+=v1
            b1GradientErr+=errF
            e=CalcError(y,a2)
            eMiniBatch+=e
         #Update w and b
         w1=w1-(eta/minibatchSize)*w1GradientErr
         w2=w2-(eta/minibatchSize)*w2GradientErr
         b1=b1-(eta/minibatchSize)*b1GradientErr
         b2=b2-(eta/minibatchSize)*b2GradientErr
         
         globalError+=eMiniBatch/minibatchSize
     globalError/=numOfMinibatchs
     print("globalError")
     print(globalError)
     print("NumOfTrue")
     print(numOfTrue)
     print("numOfFalse")
     print(numOfFalse)
     print(str(numOfTrue/600)+"%")
     #evaluation
     evalTrue=0
     evalFalse=0
     for testData in testDataTuples:
        x=testData[0].reshape(testData[0].shape[0],1)
        y=testData[1].reshape(testData[1].shape[0],1)
        z1=np.dot(w1,x)+b1
        a1=sigmoid(z1)
        z2=np.dot(w2,a1)+b2
        a2=sigmoid(z2)

        c=np.argmax(a2)
        l=np.argmax(y)
        if (c==l):
            evalTrue+=1
        else:
            evalFalse+=1
     print("evalTrue")
     print(evalTrue)
     print("evalFalse")
     print(evalFalse)
     print(str(evalTrue/100)+"%")



            


         











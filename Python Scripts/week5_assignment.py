from operator import le
import pickle
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import sys
from scipy.stats import norm
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.stats import multivariate_normal



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten,MaxPooling2D,Conv2D,Dropout,LeakyReLU,BatchNormalization
import keras

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


def class_acc(pred, gt):
    incorrect=0;
    dataLength=len(pred)
    for i in range(dataLength):
        if(pred[i]!=gt[i]):
            incorrect=incorrect+1    
    return ((dataLength-incorrect)/dataLength)*100;

def cifar10_classifier_random(X):
    Y_random_label=[]
    total_label_classes=10
    for i in range(X.shape[0]):
        Y_random_label.append(randint(0,(total_label_classes-1)))
    
    return Y_random_label;



def cifar10_classifier_1nn(x,trdata,trlabels):
    max = sys.maxsize
    Y_pred=[]
    len_sample=len(x)
    len_train=len(trdata)
    for i in range(len_sample):
        distance=max
        Y_pred.append(0)
        for j in range(len_train):
            new_distance=np.sum((x[i] - trdata[j])**2)
            if(new_distance<distance):
                distance=new_distance
                Y_pred[i]=trlabels[j]

    return Y_pred
                




def cifar10_n_color(X,n):  
    len_x=len(X)
    Images_mean = np.zeros((len_x,n,n,3))
    for i in range(Images_mean.shape[0]):
        # Resizing Images Using resize 
        img = resize(X[i], (n, n))      
        Images_mean[i] = np.array((img[:,:,0], img[:,:,1], img[:,:,2])).transpose()

    return Images_mean


#####Writing one function to calculate all the parameters
def  cifar_10_naivebayes_n_bayes_learn(X_reduced, Ytr, Ylbl, n):
    X_reduced=X_reduced.transpose(0,3,1,2)
    reds = dict()
    greens = dict()
    blues = dict()
    image = dict()


    train_data_len=len(X_reduced)
    total_class=len(Ylbl)
    dataperclass=train_data_len/total_class
    priors=[0]*total_class


    for i in range(total_class):
        reds[i]=np.array([])
        greens[i] = np.array([])
        blues[i]=np.array([])
        image[i]= np.array([])


    for i in range(len(X_reduced)):
        reds[Ytr[i]]=np.append(reds[Ytr[i]], X_reduced[i][0])
        greens[Ytr[i]] = np.append(greens[Ytr[i]], X_reduced[i][1])
        blues[Ytr[i]] = np.append(blues[Ytr[i]], X_reduced[i][2])
        image[Ytr[i]] = np.append(image[Ytr[i]], X_reduced[i])
        priors[Ytr[i]]= priors[Ytr[i]]+1
    
    covariances= np.arange(total_class*(n*n*3)*(n*n*3)).reshape(total_class,n*n*3,n*n*3).astype("float32")
    stds= np.arange(total_class*3*n*n).reshape(total_class,3*n*n).astype("float32")
    means = np.arange(total_class*3*n*n).reshape(total_class,3*n*n).astype("float32")

    for i in range(0,total_class):
        image[i] = image[i].reshape(priors[i],3*n*n)
        reds[i] = reds[i].reshape(priors[i],n,n)
        greens[i] = greens[i].reshape(priors[i],n,n)
        blues[i] = blues[i].reshape(priors[i],n,n)
        means[i] = np.concatenate((np.mean(reds[i],axis=0),np.mean(greens[i],axis=0),np.mean(blues[i],axis=0))).reshape(n*n*3)  
        stds[i] = np.concatenate((np.std(reds[i],axis=0),np.std(greens[i],axis=0),np.std(blues[i],axis=0))).reshape(n*n*3)  
        covariances[i] = np.cov(image[i],rowvar=False)


    priors_=np.array(priors)/train_data_len

    return means,stds,covariances,priors_

def cifar10_classifier_naivebayes(x,mu,sigma,p,n):
    len_mu = len(mu)
    predictions= []
    len_of_x=len(x)
    for i in range(len_of_x):
        probs= np.ones(len_mu)
        for j in range(len_mu):
            for k in range(3*n*n):
                img=x[i].transpose(2,0,1).reshape(n*n*3)
                probs[j] *= norm.pdf(img[k], mu[j][k], sigma[j][k])*priors[j]    
        predictions.append(probs.argmax())            

    return predictions

def cifar10_classifier_bayes(x,mu,sigma,p,n):
    predictions= []
    len_mu = len(mu)
    len_of_x=len(x)
    for i in range(len_of_x):
        probs= np.ones(len_mu)
        for j in range(len_mu):
            img=x[i].transpose(2,0,1).reshape(n*n*3)
            probs[j] *= multivariate_normal.logpdf(img, mu[j], sigma[j],allow_singular=True)*priors[j]
        predictions.append(probs.argmax())            
    return predictions




datadict1 = unpickle('/home/sayhelloxd/homeworkassigment3/cifar-10-python/cifar-10-batches-py/data_batch_1')
datadict2 = unpickle('/home/sayhelloxd/homeworkassigment3/cifar-10-python/cifar-10-batches-py/data_batch_2')
datadict3 = unpickle('/home/sayhelloxd/homeworkassigment3/cifar-10-python/cifar-10-batches-py/data_batch_3')
datadict4 = unpickle('/home/sayhelloxd/homeworkassigment3/cifar-10-python/cifar-10-batches-py/data_batch_4')
datadict5 = unpickle('/home/sayhelloxd/homeworkassigment3/cifar-10-python/cifar-10-batches-py/data_batch_5')

test_datadict = unpickle('/home/sayhelloxd/homeworkassigment3/cifar-10-python/cifar-10-batches-py/test_batch')
labeldict = unpickle('/home/sayhelloxd/homeworkassigment3/cifar-10-python/cifar-10-batches-py/batches.meta')


X_train = np.concatenate((datadict1["data"],datadict2["data"],datadict3["data"],datadict4["data"],datadict5["data"]))
Y_train = np.concatenate((datadict1["labels"], datadict2["labels"], datadict3["labels"], datadict4["labels"], datadict5["labels"]))
X_test=test_datadict["data"]
Y_test=test_datadict["labels"]
label_names = labeldict["label_names"]



X_train = X_train.reshape(len(X_train), 3, 32, 32).transpose(0,2,3,1).astype("float32")#.astype("uint8")
X_test = X_test.reshape(len(X_test), 3, 32, 32).transpose(0,2,3,1).astype("float32")#.astype("uint8")

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

X_tr=X_train
X_tt=X_test


print('Training data shape: ', X_train.shape)
print('Training labels shape: ', Y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)
"""
Xtr_reduced_8x8=  cifar10_n_color(X_train, 8)
Xtt_reduced_8x8= cifar10_n_color(X_test,8)
means, stds, covariances, priors=cifar_10_naivebayes_n_bayes_learn(Xtr_reduced_8x8, Y_train, label_names, 8)
tr_predictions_bayes_8x8=cifar10_classifier_bayes(Xtt_reduced_8x8,means,covariances,priors,8)
acc_bayes_8x8=class_acc(tr_predictions_bayes_8x8,Y_test)


Hard_coded them, because, it takes several minutes to run both of these two classifiers
Accuracy of Bayes: 8x8 41.71
1NN Accuraccy: 35.39
"""
acc_bayes_8x8=41.71
acc_1NN=35.39

###Converting Y_train => Y_train_hot_vectors, Y_test => Y_test_hot_vectors
Y_train_hot_vectors=np.zeros((Y_train.size, Y_train.max() + 1))
Y_train_hot_vectors[np.arange(Y_train.size), Y_train] = 1
Y_test_hot_vectors=np.zeros((Y_test.size, Y_test.max() + 1))
Y_test_hot_vectors[np.arange(Y_test.size), Y_test] = 1



print(Y_train[0:5])
print('======')

print(Y_train_hot_vectors[0:5])
print('======')
print(label_names)

X_tr,X_tt=X_tr/255.0,X_tt/255.0
model = Sequential(name = 'CNN_Cifar10')
lr=0.2
dc=1e-5
dr=0.5
epochs=50
opt = keras.optimizers.SGD(learning_rate=lr, decay=dc)
#model.add(keras.Input(shape=(32,32,3))) 
model.add(Conv2D(64, (3, 3), activation='relu'))

#model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
#model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(1e-4), padding='same', input_shape=(32, 32, 3)))

model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(dr))
model.add(Dense(10, activation='sigmoid'))

model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_tr, Y_train_hot_vectors, epochs=epochs, 
                    validation_data=(X_tt, Y_test_hot_vectors))




test_loss, test_acc = model.evaluate(X_tt,  Y_test_hot_vectors, verbose=2)


print(f'Accurarcy of Nueral Network with learning: {lr} and epochs: {epochs} =>',(test_acc*100))
print(f'Accurary of 1NN=> {acc_1NN} and bayes_8x8=>{acc_bayes_8x8}')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()


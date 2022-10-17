import pickle
from turtle import distance
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import sys


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


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


datadict = unpickle('/home/sayhelloxd/homeworkassigment3/cifar-10-python/cifar-10-batches-py/data_batch_1')
test_datadict = unpickle('/home/sayhelloxd/homeworkassigment3/cifar-10-python/cifar-10-batches-py/test_batch')
labeldict = unpickle('/home/sayhelloxd/homeworkassigment3/cifar-10-python/cifar-10-batches-py/batches.meta')


X_train = datadict["data"]
Y_train = datadict["labels"]
X_test=test_datadict["data"]
Y_test=test_datadict["labels"]
label_names = labeldict["label_names"]

#print(len(Y_train), Y_train[0], X_train[0], label_names[0], Y_test[0], Y_train[0])


X_train = X_train.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y_train = np.array(Y_train)
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y_test = np.array(Y_test)
#print('Training data shape: ', X_train.shape)
#print('Training labels shape: ', Y_train.shape)
#print('Test data shape: ', X_test.shape)
#print('Test labels shape: ', Y_test.shape)



##TESTING
"""
for i in range(len(label_names)):
    print(label_names[i], randint(0, 9))
"""   


Y_random_class=cifar10_classifier_random(X_test)

print("Point 2) Accurracy Test with same data: ",class_acc(Y_test, Y_test), "Accurracy Test with train and test data from cifar:", class_acc(Y_train, Y_test))
print("Point 3) Test data cifar10_classifier_random(X) accurary", class_acc(Y_random_class,Y_test))
print("Point 4)",class_acc(cifar10_classifier_1nn(X_test[0:5],X_train[0:10000],Y_train[0:10000]), Y_test[0:5]))


#print(X_train[2])



"""
for i in range(X_train.shape[0]):
    # Show some images randomly
    if random() > 0.999:
        plt.figure(1);
        plt.clf()
        plt.imshow(X_train[i])
        plt.title(f"Image {i} label={label_names[Y_train[i]]} (num {Y_train[i]})")
        plt.pause(1)


#X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
"""
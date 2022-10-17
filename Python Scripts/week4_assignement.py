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

Xtr=X_train

X_train = X_train.reshape(len(X_train), 3, 32, 32).transpose(0,2,3,1).astype("float32")#.astype("uint8")
X_test = X_test.reshape(len(X_test), 3, 32, 32).transpose(0,2,3,1).astype("float32")#.astype("uint8")

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

print('Training data shape: ', X_train.shape)
print('Training labels shape: ', Y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)

#Xtr_red=  cifar10_n_color(X_train[0:100],2)
#Xtt_red = cifar10_n_color(X_test[0:100],2)
#print('Mean of RGB for First Train and Test Image converting [0][32][32][3] => [0][1][1][3], Training =>', Xtr_red.transpose(0,3,2,1)[0], ", Test =>", Xtt_red.transpose(0,3,2,1)[0], "\n\n\n")
#means, stds, covariances, priors=cifar_10_naivebayes_n_bayes_learn(Xtr_red, Y_train, label_names, 2)
#print("Means[0] =>", means[0], ", stds[0] =>", stds[0],", covariances[0] =>", covariances[0], ", priors[0] =>", priors[0], "\n\n\n")





Xtr_reduced=  cifar10_n_color(X_train, 1)
Xtt_reduced = cifar10_n_color(X_test, 1)


###Calculating one function to get paramters
means, stds, covariances, priors=cifar_10_naivebayes_n_bayes_learn(Xtr_reduced, Y_train, label_names, 1)


print("Means[0] =>", means[0], ", stds[0] =>", stds[0],", covariances[0] =>", covariances[0], ", priors[0] =>", priors[0], "\n\n\n")


tr_predictions_naivebayes=cifar10_classifier_naivebayes(Xtt_reduced,means,stds,priors,1)
tr_predictions_bayes=cifar10_classifier_bayes(Xtt_reduced,means,covariances,priors,1)
print("Accuracy of NaiveBayes:", class_acc(tr_predictions_naivebayes,Y_test))
print("Accuracy of Bayes:", class_acc(tr_predictions_bayes,Y_test))


Xtr_reduced=  cifar10_n_color(X_train, 1)
Xtt_reduced = cifar10_n_color(X_test, 1)
means, stds, covariances, priors=cifar_10_naivebayes_n_bayes_learn(Xtr_reduced, Y_train, label_names, 1)
tr_predictions_naivebayes=cifar10_classifier_naivebayes(Xtt_reduced,means,stds,priors,1)
tr_predictions_bayes=cifar10_classifier_bayes(Xtt_reduced,means,covariances,priors,1)
acc_naive_bayes_1x1=class_acc(tr_predictions_naivebayes,Y_test)
acc_bayes_1x1=class_acc(tr_predictions_bayes,Y_test)
print("Accuracy of NaiveBayes: 1x1", acc_naive_bayes_1x1)
print("Accuracy of Bayes: 1x1", acc_bayes_1x1)


#############2x2####################
Xtr_reduced_2x2=  cifar10_n_color(X_train, 2)
Xtt_reduced_2x2 = cifar10_n_color(X_test,2)
means, stds, covariances, priors=cifar_10_naivebayes_n_bayes_learn(Xtr_reduced_2x2, Y_train, label_names, 2)
tr_predictions_naivebayes_2x2=cifar10_classifier_naivebayes(Xtt_reduced_2x2,means,stds,priors,2)
tr_predictions_bayes_2x2=cifar10_classifier_bayes(Xtt_reduced_2x2,means,covariances,priors,2)
acc_naive_bayes_2x2=class_acc(tr_predictions_naivebayes_2x2,Y_test)
acc_bayes_2x2=class_acc(tr_predictions_bayes_2x2,Y_test)
print("Accuracy of NaiveBayes: 2x2", acc_naive_bayes_2x2)
print("Accuracy of Bayes: 2x2", acc_bayes_2x2)
#############4x4###############
Xtr_reduced_4x4=  cifar10_n_color(X_train, 4)
Xtt_reduced_4x4= cifar10_n_color(X_test,4)
means, stds, covariances, priors=cifar_10_naivebayes_n_bayes_learn(Xtr_reduced_4x4, Y_train, label_names, 4)
tr_predictions_naivebayes_4x4=cifar10_classifier_naivebayes(Xtt_reduced_4x4,means,stds,priors,4)
tr_predictions_bayes_4x4=cifar10_classifier_bayes(Xtt_reduced_4x4,means,covariances,priors,4)
acc_naive_bayes_4x4=class_acc(tr_predictions_naivebayes_4x4,Y_test)
acc_bayes_4x4=class_acc(tr_predictions_bayes_4x4,Y_test)
print("Accuracy of NaiveBayes: 4x4", acc_naive_bayes_4x4)
print("Accuracy of Bayes: 4x4", acc_bayes_4x4)
#############8x8###############
Xtr_reduced_8x8=  cifar10_n_color(X_train, 8)
Xtt_reduced_8x8= cifar10_n_color(X_test,8)
means, stds, covariances, priors=cifar_10_naivebayes_n_bayes_learn(Xtr_reduced_8x8, Y_train, label_names, 8)
tr_predictions_naivebayes_8x8=cifar10_classifier_naivebayes(Xtt_reduced_8x8,means,stds,priors,8)
tr_predictions_bayes_8x8=cifar10_classifier_bayes(Xtt_reduced_8x8,means,covariances,priors,8)
acc_naive_bayes_8x8=class_acc(tr_predictions_naivebayes_8x8,Y_test)
acc_bayes_8x8=class_acc(tr_predictions_bayes_8x8,Y_test)
print("Accuracy of NaiveBayes: 8x8", acc_naive_bayes_8x8)
print("Accuracy of Bayes: 8x8", acc_bayes_8x8)

""""
Accuracy of NaiveBayes: 1x1 19.52
Accuracy of Bayes: 1x1 24.7
Accuracy of NaiveBayes: 2x2 22.5
Accuracy of Bayes: 2x2 30.97
Accuracy of NaiveBayes: 4x4 26.040000000000003
Accuracy of Bayes: 4x4 40.22
Accuracy of NaiveBayes: 8x8 10.0
Accuracy of Bayes: 8x8 41.71
"""
##### Hardcoded values after getting output from the previous code
#y_b_graph = acc_bayes_1x1, acc_bayes_2x2, acc_bayes_4x4 , acc_bayes_8x8 
#y_nb_graph = acc_naive_bayes_1x1, acc_naive_bayes_2x2,acc_naive_bayes_4x4,acc_naive_bayes_8x8
y_b_graph = 24.7, 30.97, 40.22 , 41.71 
y_nb_graph = 19.52, 22.5,26.040000000000003,10.0
x_b_graph = np.array([1,2,4,8])
x_nb_graph = np.array([1,2,4,8])


plt_xticks = ['1x1','2x2','4x4','8x8']
plt.xticks(x_b_graph, plt_xticks)
plt.plot(x_b_graph, y_b_graph, marker='.',markerfacecolor='red', markersize=9, color='orange', linewidth=3, label = "Bayes")
plt.legend()
plt.grid(True)
plt.axis([0,20,0,60])


for i,j in zip(x_b_graph,y_b_graph):
    plt.annotate(str(j),xy=(i-0.25,j+4))
    
    
plt.plot(x_nb_graph, y_nb_graph, marker='.',markerfacecolor='blue', markersize=9, color='cyan', linewidth=3, label = 'Naive Bayes')
plt.legend()
for i,j in zip(x_nb_graph,y_nb_graph):
    plt.annotate(str(j),xy=(i-0.25,j-5))

plt.xlabel('Image Size')
plt.ylabel('Acc. (%)')
plt.show()

Xtr_reduced_9x9=  cifar10_n_color(X_train, 9)
Xtt_reduced_9x9= cifar10_n_color(X_test,9)
means, stds, covariances, priors=cifar_10_naivebayes_n_bayes_learn(Xtr_reduced_9x9, Y_train, label_names, 9)
tr_predictions_naivebayes_9x9=cifar10_classifier_naivebayes(Xtt_reduced_9x9,means,stds,priors,9)
tr_predictions_bayes_9x9=cifar10_classifier_bayes(Xtt_reduced_9x9,means,covariances,priors,9)
acc_naive_bayes_9x9=class_acc(tr_predictions_naivebayes_9x9,Y_test)
acc_bayes_9x9=class_acc(tr_predictions_bayes_9x9,Y_test)
print("Accuracy of NaiveBayes: 9x9", acc_naive_bayes_9x9)
print("Accuracy of Bayes: 9x9", acc_bayes_9x9)


#############16x16###############
Xtr_reduced_16x16=cifar10_n_color(X_train, 16)
Xtt_reduced_16x16=cifar10_n_color(X_test,16)
print("resize done")
means, stds, covariances, priors=cifar_10_naivebayes_n_bayes_learn(Xtr_reduced_16x16, Y_train, label_names, 16)
print("parameters done")

tr_predictions_bayes_16x16=cifar10_classifier_bayes(Xtt_reduced_16x16,means,covariances,priors,16)
acc_bayes_16x16=class_acc(tr_predictions_bayes_16x16,Y_test)
print("Accuracy of Bayes: 6x16", acc_bayes_16x16)


#############32x32###############
Xtr_reduced_32x32=cifar10_n_color(X_train, 32)
Xtt_reduced_32x32=cifar10_n_color(X_test,32)
means, stds, covariances, priors=cifar_10_naivebayes_n_bayes_learn(Xtr_reduced_32x32, Y_train, label_names, 32)
tr_predictions_bayes_32x32=cifar10_classifier_bayes(Xtt_reduced_16x16,means,covariances,priors,32)
acc_bayes_32x32=class_acc(tr_predictions_bayes_32x32,Y_test)
print("Accuracy of Bayes: 32x32", acc_bayes_32x32)




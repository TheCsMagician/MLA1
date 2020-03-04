# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:47:37 2018

@author: zainu
"""

import numpy as np
import numpy.random as rnd
import math
import matplotlib.pyplot as plt
import pickle 
import sklearn.linear_model as lin
import time


print "\n\nQuestion 1"
print "----------"
print "\nQuestion 1(a):"
a = rnd.rand(4,3)
print a
print "\nQuestion 1(b):"
x = rnd.rand(4,1)
print x
print "\nQuestion 1(c):"
b = np.reshape(a,(2,6))
print b
print "\nQuestion 1(d):"
c = a + x
print c
print "\nQuestion 1(e):"
y = np.reshape(x,(4))
print y
print "\nQuestion 1(f):"
a[:,0] = np.transpose(y)
print a
print "\nQuestion 1(g):"
a[:,0] = a[:,2] + np.transpose(y)
print a
print "\nQuestion 1(h):"
print [a[:,0],a[:,1]]
print "\nQuestion 1(i):"
print [a[1,:],a[3,:]]
print "\nQuestion 1(j):"
l = a[0,:]+a[1,:]+a[2,:] + a[3,:]
print l
print "\nQuestion 1(k):"
k = [np.max(a[0,:]),np.max(a[1,:]),np.max(a[2,:]),np.max(a[3,:])]
print k
print "\nQuestion 1(l):"
print np.mean(a)
print "\nQuestion 1(m):"
l = np.multiply(a,a)
print np.log(l)
print "\nQuestion 1(n):"
print np.matmul(np.transpose(a),x)

print "\n\nQuestion 2"
print "----------"
print "\nQuestion 2(a):"
b = rnd.rand(2,2)

def helper(A):
    b = np.zeros(np.shape(A))
    print b
    c = np.shape(A)
    for row in range(c[0]):
        for col in range(c[0]):
            for k in range(c[0]):
                b[row][col] += A[row][k] * A[k][col]
    return b
def cube(A):
    b = helper(A)
    return helper(b)


print "\nQuestion 2(b):"

def mymeasure(N):
    b = rnd.rand(N,N)
    c= rnd.rand(N,N)
    cube1 = np.matmul(np.matmul(b,b),b)
    print "time.time(): %f " %  time.time()
    cube2 = cube(c)
    print "time.time(): %f " %  time.time()
    print "Max of magnitude: %f" % np.max(np.abs(np.subtract(cube1,cube2)))
mymeasure(200)
#mymeasure(2000)


print "\n\nQuestion 4"
print "----------"
print "\nQuestion 4(a):"

with open("data1.pickle","rb") as f:
    dataTrain,dataTest = pickle.load(f)


def kernelMatrix(X,S,sigma):
    n = np.shape(X)
    m = np.shape(S)
    k = np.zeros((n[0],m[0]+1))
    k[:,0] = 1    
    for i in range(1,m[0]+1):
        k[:,i] = X
    
    for m in range(1,m[0]+1):
        k[:,m] = np.exp(np.divide(np.negative(np.square(k[:,m] - S[m-1])),2*(sigma**2)))
    
    return k

X = [3,2]
S = [1,2]
print kernelMatrix(X,S,1)
print "\nQuestion 4(b):"

def plotBasis(S,sigma):
    X = np.linspace(0.0,1.0, num=1000)
    Y = kernelMatrix(X,S[:,0],sigma)
    plt.plot(X,Y)
    plt.title("Training Set")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

print plotBasis(dataTrain[0:5],0.2)
print plotBasis(dataTrain[0:5],0.1)

print dataTrain
print dataTrain[:,0]
print "\nQuestion 4(c):"
def myfit(S,sigma):
     K = kernelMatrix(dataTrain[:,0],S[:,0],sigma)
     w = np.linalg.lstsq(K,dataTrain[:,1],rcond=None)[0]
     err = 0
     K_t = np.transpose(K)
     for n in range(len(dataTrain)):
         err += np.square(dataTrain[n,1]-np.matmul(w,K_t[:,n]))
     err_train = err/len(dataTrain)
     err = 0
     err_test = 0
     K = kernelMatrix(dataTest[:,0],S[:,0],sigma)
     #w = np.linalg.lstsq(K,dataTest[:,1],rcond=None)[0]
     K_t = np.transpose(K)
     for n in range(len(dataTest)):
         err += np.square(dataTest[n,1]-np.matmul(w,K_t[:,n]))
     err_test = err/len(dataTest)
     return w,err_train,err_test


    


print "\nQuestion 4(d):"
def plotY(w,S,sigma):
    X = np.linspace(0.0,1.0, num=1000)
    K = kernelMatrix(X,S[:,0],sigma)
    Y = np.matmul(w,np.transpose(K))
    plt.plot(S[:,0],S[:,1],'bo')
    plt.plot(X,Y,'r')
    plt.ylim((-15,15))
    return plt

print "\nQuestion 4(e):"

p =myfit(dataTrain[0:5],0.2)
plotY(p[0],dataTrain[0:5],0.2).title("Question 4(e): the fitted function (5 basis functions)")
plotY(p[0],dataTrain[0:5],0.2).show()


print "\nQuestion 4(f):"

def bestM(sigma):
    
    p = myfit(dataTrain[0:0],sigma)
    plotY(p[0],dataTrain[0:0],sigma).subplot(4,4,1)
    for i in range (15):
        p = myfit(dataTrain[0:i],sigma)
        plotY(p[0],dataTrain[0:i],sigma).subplot(4,4,i+2)
    p = myfit(dataTrain[0:15],sigma)
    plotY(p[0],dataTrain[0:15],sigma).subplot(4,4,15)
    plt.suptitle("Question 4(f): best-fitting functions with 0-15 basis functions")
    plt.show()
    err_training = []
    err_testing = []
    for i in range (15):
        p = myfit(dataTrain[0:i],sigma)
        err_training.append(p[1])
        err_testing.append(p[2])
    M = range(15)
    plt.plot(M,err_training,'b')
    plt.plot(M,err_testing,'r')
    plt.xlabel("M")
    plt.ylabel("Error values")
    plt.ylim((0,250))
    plt.title("Question 4(f): training and test error")
    plt.show()    
    index_min = np.argmin(err_testing)
    p = myfit(dataTrain[0:index_min],sigma)
    plotY(p[0],dataTrain[0:index_min],sigma).title("Question 4(f): best-fitting function (M basis functions)")
    plotY(p[0],dataTrain[0:index_min],sigma).show()
    print "M =",index_min,"w =",p[0]
    print "err_train =",p[1],"err_test =",p[2]
    
bestM(0.2)

print "\nQuestion 5:"
with open("data2.pickle","rb") as f:
    dataVal,dataTest = pickle.load(f)
print "----------"
print "\nQuestion 5(a):"

def regFit(S,sigma,alpha):

    K = kernelMatrix(dataTrain[:,0],S[:,0],sigma)
    K1 = kernelMatrix(dataVal[:,0],S[:,0],sigma)
    ridge = lin.Ridge(alpha)
    ridge.fit(K,S[:,1])
    w = ridge.coef_
    w[0] = ridge.intercept_
    err = 0
    K_t = np.transpose(K)
    K1_t = np.transpose(K1)
    for n in range(len(dataVal)):
         err += np.square(dataVal[n,1]-np.matmul(w,K1_t[:,n]))
    err_val = err/len(dataVal)
    err = 0
    
    
    for n in range(len(dataTrain)):
         err += np.square(dataTrain[n,1]-np.matmul(w,K_t[:,n]))
    err_train = err/len(dataTrain)
    return w,err_train,err_val
    

print "\nQuestion 5(b):"
p = regFit(dataTrain[0:15],0.2,1)
a = plotY(p[0],dataTrain[0:15],0.2)
a.title("Question 5(b): the fitted function (alpha=1)")
a.show()

print "\nQuestion 5(c):"
def bestAlpha(S,sigma):
    a = range(-12,4)
    for i in range(14):
        a[i] = 10**a[i]
    for i in range(len(a)):
        r = regFit(S,sigma,a[i])
        plotY(r[0],S,sigma).subplot(4,4,i+1)
    plt.suptitle("Question 5(c): best-fitting functions for log(alpha) = -12,-11,...,1,2,3")
    plt.show()
    err_training = []
    err_val = []
    for i in range (len(a)):
        p = regFit(S,sigma,a[i])
        err_training.append(p[1])
        err_val.append(p[2])
    plt.semilogx(a,err_training,'b')
    plt.semilogx(a,err_val,'r')
    plt.xlabel("alpha")
    plt.ylabel("error")
    plt.title("Question 5(c): training and validation error")
    plt.show()
    index_min = np.argmin(err_val)
    r = regFit(S,sigma,a[index_min])
    plotY(r[0],S,sigma).title("Question 5(c): best-fitting function alpha= {0}".format(a[index_min]))
    plotY(r[0],S,sigma).show()
    K = kernelMatrix(dataTest[:,0],S[:,0],sigma)
    err = 0
    K_t = np.transpose(K)
    for n in range(len(dataTest)):
         err += np.square(dataTest[n,1]-np.matmul(r[0],K_t[:,n]))
    err_test = err/len(dataTest)
    print "alpha =", a[index_min], "w =", r[0]
    print "training error =",r[1],"training validation =",r[2],"test error =",err_test
    
bestAlpha(dataTrain[0:15],0.2)


print "\n\nQuestion 6"
print "----------"
print "\nQuestion 6(a):"
ran = dataVal
np.random.shuffle(ran)
plt.plot(ran[:,0],ran[:,1],'ro')
plt.title("Question 6(a): Training data for Question 6")
plt.show()
print "\nQuestion 6(b):"

    

def cross_val(K,S,sigma,alpha,X,Y):
    k_err_train = []
    k_err_val = []
    K_X = np.split(X,K)
    K_Y = np.split(Y,K)
    for i in range(K):
        P = kernelMatrix(K_X[i],S[:,0],sigma)
        ridge = lin.Ridge(alpha)
        ridge.fit(P,K_Y[i])
        w = ridge.coef_
        w[0] = ridge.intercept_
        P = kernelMatrix(dataVal[:,0],S[:,0],sigma)
        err = 0
        K1_t = np.transpose(P)
        for n in range(len(dataVal)):
            err += np.square(dataVal[n,1]-np.matmul(w,K1_t[:,n]))
        err_val = err/len(dataVal)
        P = kernelMatrix(dataTrain[:,0],S[:,0],sigma)
        err = 0
        K1_t = np.transpose(P)
        for n in range(len(dataTrain)):
            err += np.square(dataTrain[n,1]-np.matmul(w,K1_t[:,n]))
        err_train = err/len(dataTrain)
        k_err_val.append(err_val)
        k_err_train.append(err_train)
    return k_err_train, k_err_val

print "\nQuestion 6(c):"
w = cross_val(5,dataVal[0:10],0.2,1,ran[:,0],ran[:,1])
fold = range(1,6)
plt.plot(fold,w[0],'b')
plt.plot(fold,w[1],'r')
plt.title("Question 6(c): training and validation errors during cross validation")
plt.xlabel("fold")
plt.ylabel("error")
plt.show()        
print "\nQuestion 6(d):"

def Average(lst): 
    return sum(lst) / len(lst)
def bestAlphaCV(K,S,sigma,X,Y):
    mean_v = []
    mean_t = []
    a = range(-11,5)
    for i in range(len(a)):
        a[i] = 10**a[i]
    for i in range(len(a)):
        w = cross_val(K,S,sigma,a[i],X,Y)
        mean_v.append(Average(w[0]))
        mean_t.append(Average(w[1]))
    plt.semilogx(a,mean_t,'b')
    plt.semilogx(a,mean_v,'r')
    plt.xlabel("alpha")
    plt.ylabel("error")
    plt.title("Question 6(d): training and validation error")
    plt.show()
    index_min = np.argmin(mean_v)
    r = regFit(S,sigma,a[index_min])
    err = 0
    P = kernelMatrix(dataTest[:,0],S[:,0],sigma)
    K_t = np.transpose(P)
    for n in range(len(dataTest)):
         err += np.square(dataTest[n,1]-np.matmul(r[0],K_t[:,n]))
    err_test = err/len(dataTest)
    err = 0
    P = kernelMatrix(ran[:,0],S[:,0],sigma)
    K_t = np.transpose(P)
    for n in range(len(ran)):
         err += np.square(ran[n,1]-np.matmul(r[0],K_t[:,n]))
    err_train = err/len(ran)
    plotY(r[0],S,sigma).title("Question 6(d): best-fitting function (alpha = {0})".format(a[index_min]))
    plotY(r[0],S,sigma).xlabel("x")
    plotY(r[0],S,sigma).ylabel("y")
    plotY(r[0],S,sigma).show()
    print "w =", r[0],"alpha =",a[index_min]
    print "training = ",err_train,"Mean validation = ",Average(mean_v),"testing = ",err_test
   
bestAlphaCV(5,dataVal[0:15],0.2,ran[:,0],ran[:,1])

print "\n\nQuestion 7"
print "----------"
print "I don't know"
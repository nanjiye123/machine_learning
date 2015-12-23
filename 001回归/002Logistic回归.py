# -*- coding: utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt
import math

def loadData(fileName):
    xMat = []
    yMat = []
    fr = open(fileName)
    for line in fr.readlines():
        tmp = []
        curLine = map(float, line.strip().split(','))#strip()去掉\n
        tmp.append( curLine[0] )
        tmp = feature_whole(curLine[1], curLine[2], 2) #共2维
        xMat.append( tmp )
        yMat.append( float(curLine[3]) )
    return mat(xMat), mat(yMat).T

#将xMat变成[1,x1,x2,x1^2,x1*x2,x2^2]
def feature_whole(x1, x2, e):
    d = []
    for n in range(1,e+1):
        for i in range(n+1):
            d.append(pow(x1, n-i) * pow(x2, i))#x的y次方
    return d


#Logistic函数
def h(x):
    if(x > 500):
        return 1.0
    if(x < -500):
        return 0.0
    return 1.0/(1.0+math.exp(-x)) #Logistic函数

#批处理梯度下降法的Logistic回归
def logistic_regression1(xMat, yMat):
    alpha = 0.001    #设置的参数
    lamda = 0.00001  #设置的参数
    timesValue = 5000   #设置的参数
    N = xMat.shape[1] #数据的维度
    w = mat([0.0 for x in range(N)])
    w2 = mat([0.0 for x in range(N)])
    #print xMat.shape, yMat.shape, w.shape
    
    for times in range(timesValue): #迭代次数
        for i in range(yMat.shape[0]):
            w2 = w2 + alpha * (yMat[i] - h(xMat[i,:] * w.T))*xMat[i,:] + lamda*w
        w = w2
        print times,w
    return w, xMat, yMat

#画图主程序
def showImage(w, xMat, yMat):
    #数据点说明x1==1，用x2和x3画图
    rx1 = []#第一类点红色 y=0
    rx2 = []
    bx1 = []#第二类点蓝色 y=1
    bx2 = []
    rx1_hat = []
    rx2_hat = []
    bx1_hat = []
    bx2_hat = []
    for i in range(yMat.shape[0]): #画点
        if yMat[i, 0] == 0:
            rx1.append(xMat[i, 1])
            rx2.append(xMat[i, 2])
        else:
            bx1.append(xMat[i, 1])
            bx2.append(xMat[i, 2])
    for i in range(yMat.shape[0]):
        if h(xMat[i,:]*w.T) <0.5:
            rx1_hat.append(xMat[i, 1])
            rx2_hat.append(xMat[i, 2])
        else:
            bx1_hat.append(xMat[i, 1])
            bx2_hat.append(xMat[i, 2])


    plt.plot(rx1, rx2, 'ro', markersize=10)
    plt.plot(bx1, bx2, 'bo', markersize=10)
    plt.plot(rx1_hat, rx2_hat, 'ro', markersize=5)
    plt.plot(bx1_hat, bx2_hat, 'bo', markersize=5)
    plt.grid(True)
    plt.show()

def aliyun_test(w, xMat, yMat):
    result = 0
    for i in range(yMat.shape[0]):
        if h(xMat[i,:]*w.T)<0.5 and yMat[i,0]==0:
            result = result + 1
        if h(xMat[i,:]*w.T)>=0.5 and yMat[i,0]==1:
            result = result + 1
    print result
    return result


if __name__ == "__main__":
    #数据格式说明 共100个数据
    #x1,x2,x3,y = (1, x2, x3 , 1 or 0)
    #xArr 100*3， yArr 1*100
    xMat, yMat = loadData("data002.txt")
    w1, xMat1, yMat1 = logistic_regression1(xMat, yMat)
    #showImage(w1, xMat1, yMat1)
    aliyun_test(w1,xMat1,yMat1)



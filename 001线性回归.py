# -*- coding: utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt

def loadData(fileName):
    xMat = []
    yMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')#strip()去掉\n
        tmp = []
        tmp.append( float(curLine[0]) )
        tmp.append( float(curLine[1]) )
        xMat.append( tmp )
        yMat.append( float(curLine[2]) )
    return xMat, yMat

#标准的线性回归画图
def drawStandRegres(yHat, xMat, yMat):
    fig = plt.figure()#轮廓
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat[:,0].flatten().A[0])#散点图
    ax.plot(xMat[:,1],yHat)
    plt.show()
    return

#标准的线性回归，无迭代，函数未优化
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yHat = mat([])
    loss = 0.0
    #print yMat.shape

    if linalg.det(xMat.T * xMat) == 0.0:#判断矩阵是否不可逆
        print "The matix cannot do inverse"
        return
    w = (xMat.T * xMat).I * xMat.T * yMat
    yHat = xMat * w

    #计算loss函数
    for i in range(yMat.shape[0]):
        loss = loss + (yHat[i,0] - yMat[i,0])**2
    return yHat,xMat,yMat,loss

#批处理梯度下降法的线性回归
def gradientDescent1(xArr, yArr):
    alpha = 0.001    #设置的参数
    timesValue = 500  #设置的参数
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yHat = mat([0.0 for x in range(yMat.shape[0])]).T
    N = 2 #数据的维度
    w = mat([0.0 for x in range(N)])
    w2 = mat([0.0 for x in range(N)])
    loss = 0.0
    lossValue = mat([0.0 for x in range(timesValue)])
    
    for times in range(timesValue): #迭代次数
        for i in range(yMat.shape[0]):
            w2 = w2 + alpha * (yMat[i] - xMat[i,:] * w.T) * xMat[i,:]
        w = w2
        #计算loss函数
        loss = 0.0
        yHat = xMat * w.T  
        for i in range(yMat.shape[0]):
            loss = loss + (yHat[i,0] - yMat[i,0])**2
        lossValue[0,times] = loss
        print times, loss       
    return yHat, xMat, yMat, lossValue

#随机梯度下降法的线性回归
def gradientDescent2(xArr, yArr):
    alpha = 0.001    #设置的参数
    timesValue = 500  #设置的参数
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yHat = mat([0.0 for x in range(yMat.shape[0])]).T
    N = 2 #数据的维度
    w = mat([0.0 for x in range(N)])
    loss = 0.0
    lossValue = mat([0.0 for x in range(timesValue)])
    
    for times in range(timesValue): #迭代次数
        for i in range(yMat.shape[0]):
            w = w + alpha * (yMat[i] - xMat[i,:] * w.T) * xMat[i,:]
        #计算loss函数
        loss = 0.0
        yHat = xMat * w.T  
        for i in range(yMat.shape[0]):
            loss = loss + (yHat[i,0] - yMat[i,0])**2
        lossValue[0,times] = loss
        print times, loss     
    return yHat, xMat, yMat, lossValue



if __name__ == "__main__":
    xArr, yArr = loadData("001.txt")
    yHat1,xMat1,yMat1,lossValue1 = standRegres(xArr, yArr)
    #drawStandRegres(yHat1, xMat1, yMat1)
    yHat2, xMat2, yMat2,lossValue2 = gradientDescent1(xArr, yArr)
    #drawStandRegres(yHat2, xMat2, yMat2)
    yHat3, xMat3, yMat3,lossValue3 = gradientDescent2(xArr, yArr)
    #drawStandRegres(yHat3, xMat3, yMat3)

    fig1 = plt.figure(1)
    ax11 = fig1.add_subplot(111)
    ax11.scatter(xMat1[:,1].flatten().A[0], yMat1[:,0].flatten().A[0])#散点图
    ax11.plot(xMat1[:,1],yHat1)
    print lossValue1

    fig2 = plt.figure(2)
    ax21 = fig2.add_subplot(121)
    ax21.scatter(xMat2[:,1].flatten().A[0], yMat2[:,0].flatten().A[0])
    ax21.plot(xMat2[:,1],yHat2)
    ax22 = fig2.add_subplot(122)
    ax22.plot(mat([x for x in range(lossValue2.shape[1])]), lossValue2, 'ro')
    print lossValue2[0,-1]
    
    fig3 = plt.figure(3)
    ax31 = fig3.add_subplot(121)
    ax31.scatter(xMat3[:,1].flatten().A[0], yMat3[:,0].flatten().A[0])
    ax31.plot(xMat3[:,1],yHat3)
    ax32 = fig3.add_subplot(122)
    ax32.plot(mat([x for x in range(lossValue3.shape[1])]), lossValue3, 'ro')
    print lossValue3[0,-1]
    plt.show()

    



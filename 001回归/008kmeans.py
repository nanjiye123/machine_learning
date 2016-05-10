# -*- coding: utf-8 -*-

#k均值 和 二分k均值
from numpy import *
import pylab as pl

def loadData(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float, curLine)
		dataMat.append(fltLine)
	return mat(dataMat) #80*2


def distEclud(vecA, vecB):#计算欧式距离
	return sqrt( sum(power(vecA - vecB, 2)) )


def randCent(dataSet, k):#构建一个包含k个随机质心的集合
	#dataSet是dataMat 80*2
	n = shape(dataSet)[1]            #n是维度或列数
	centroids = mat( zeros((k,n)) )  #一行是一个点
	for j in range(n):               #对每一个维度求值
		minJ = min( dataSet[:,j] )   #找到所有点中第j维的最小值
		rangeJ = float( max(dataSet[:,j]) - minJ ) #求最大值和最小值的差值
		centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1)) #随机生成值为0-1的列向量
	return centroids #返回k行2列矩阵


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
	#输入：数据集，簇的数目，计算距离的函数（欧式距离），创建初始质心的函数
	m = shape(dataSet)[0]
	clusterAssment = mat( zeros((m,2)) )
	#创建簇分配结果矩阵，一列记录簇索引值，一列记录当前点到质心的距离
	centroids = createCent(dataSet, k)#创建k个初始质心 k*2
	clusterChanged = True
	#========================================================================
	#迭代过程：计算质心--分配--重新计算
	while clusterChanged:
		clusterChanged = False
		for i in range(m): #遍历所有数据点，分配给距离这个点最近的质心
			minDist = inf; minIndex = -1
			for j in range(k): #遍历所有质心
				distJI = distMeas(centroids[j,:],dataSet[i,:]) #计算这个点到每一个质心的距离
				if distJI < minDist:
					minDist = distJI; minIndex = j #minIndex标记k个质心的标号
			if clusterAssment[i,0] != minIndex: #如果任何一个点的簇分配结果改变，标志更新继续迭代
				clusterChanged = True
			clusterAssment[i,:] = minIndex, minDist**2 #簇分配结果矩阵
		#print centroids
		for cent in range(k): #遍历所有质心，更新取值
			ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] #数组过滤，获得这个簇中的所有点的行号
			#ptsInClust存放点的坐标
			centroids[cent,:] = mean(ptsInClust, axis=0) #计算该簇内所有点的均值，axis=0沿矩阵列方向计算均值，变成新的质心
	#=========================================================================
	return centroids, clusterAssment #返回质心记录矩阵和分配结果矩阵

def drawpict(dataMat, centroids, clustAssment, k):#只适合k=4的情况
	for i in range(shape(dataMat)[0]):
		if clustAssment[i,0] == 0:
			pl.plot( dataMat[i,0], dataMat[i,1], 'ro')
		elif clustAssment[i,0] == 1:
			pl.plot( dataMat[i,0], dataMat[i,1], 'bo')
		elif clustAssment[i,0] == 2:
			pl.plot( dataMat[i,0], dataMat[i,1], 'go')
		elif clustAssment[i,0] == 3:
			pl.plot( dataMat[i,0], dataMat[i,1], 'yo')
	for j in range(k):
		pl.plot( centroids[j,0], centroids[j,1], 'k*' )
	pl.xlabel('x1')
	pl.ylabel('x2')
	pl.show()


#二分k均值算法
def biKmeans(dataSet, k, distMeas=distEclud):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2)))
	#创建簇分配结果矩阵，一列记录簇索引值，一列记录当前点到质心的距离平方
	centroid0 = mean(dataSet, axis=0).tolist()[0] #取列表的第0行
	centList =[centroid0]  #创建一个列表，初始值有一个质心
	for j in range(m):     #计算初始误差
		clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
	while (len(centList) < k): #对簇不停的划分，直到得到想要的簇数目为止
		lowestSSE = inf        #设置误差平方和为无穷大
		#==================================================================
		for i in range(len(centList)):
			ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:] #获得簇i中的所有数据点
			centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) #对一个簇进行二分kmeans
			sseSplit = sum(splitClustAss[:,1]) #将划分后的误差相加，得到误差平方和SSE
			sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) #求其他簇的SSE
			print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
			if (sseSplit + sseNotSplit) < lowestSSE:
				bestCentToSplit = i         #i是被二分的簇的质心序号
				bestNewCents = centroidMat  #二分后的两个质心序号
				bestClustAss = splitClustAss.copy()
				lowestSSE = sseSplit + sseNotSplit
		#==================================================================
		bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)   #将刚才划分的0,1簇序号改到原位置和结尾
		bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
		print 'the bestCentToSplit is: ',bestCentToSplit
		print 'the len of bestClustAss is: ', len(bestClustAss)
		centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] #存放质心坐标的列表更新
		centList.append(bestNewCents[1,:].tolist()[0])
		clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss #更新SSE
	return mat(centList), clusterAssment



if __name__ == "__main__":
	#k均值================================================
	dataMat = loadData('data008.txt')#返回的是列表 80*2
	centroids, clustAssment = kMeans(dataMat, 4) #分为4个簇
	#度量聚类效果的指标是SSE误差平方和，值越小表示数据点越接近质心，聚类效果好
	drawpict(dataMat, centroids, clustAssment, 4)
	#====================================================


	#二分k均值===========================================
	dataMat2 = loadData('data008_2.txt')
	centroids, clustAssment = kMeans(dataMat2, 3)
	drawpict(dataMat2, centroids, clustAssment, 3)
	centList, clusterAssment2 = biKmeans(dataMat2, 3)
	drawpict(dataMat2, centList, clusterAssment2, 3)
	#====================================================


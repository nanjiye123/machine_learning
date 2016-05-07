# -*- coding: utf-8 -*-
from numpy import *
from numpy import linalg as la

#基于物品相似度的推荐引擎


def loadExData():
	return[[4, 4, 0, 2, 2],
	       [4, 0, 0, 3, 3],
		   [4, 0, 0, 1, 1],
		   [1, 1, 1, 2, 0],
		   [2, 2, 2, 0, 0],
		   [5, 5, 5, 0, 0],
		   [1, 1, 1, 0, 0]]

def loadExData2():
    return[[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
	       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
		   [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
		   [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
		   [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
		   [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
		   [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
		   [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
		   [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
		   [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
		   [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]


#inA,inB是列向量
def eulidSim(inA, inB): #欧式距离，归一化
	return 1.0 / (1.0 + la.norm(inA-inB))

#inA,inB是列向量
def pearsSim(inA, inB): #皮尔逊相关系数
	if len(inA) < 3: #不存在3个或更多的点，则此时两个向量完全相关
		return 1.0
	#归一化到(0,1)
	return 0.5 + 0.5 * corrcoef(inA, inB, rowvar = 0)[0][1]

#inA,inB是列向量
def cosSim(inA, inB): #余弦相似度
	num = float(inA.T * inB)
	denom = la.norm(inA) * la.norm(inB)
	#归一化到(0,1)
	return 0.5 + 0.5 * (num/denom)

#根据相似度，计算用户对物品的估计评分值
def standEst(dataMat, user, item, simMeas):
	#数据矩阵，用户编号，物品编号，相似度计算方法
	n = shape(dataMat)[1] #列数，即物品数量
	simTotal = 0.0    #用于计算估计评分值
	ratSimTotal = 0.0 #用于计算估计评分值
	for j in range(n): #对该用户评过分的物品进行遍历
		userRating = dataMat[user, j]
		if userRating == 0: continue #评分为0,则是没有评分，跳过
		overLap = nonzero(logical_and(dataMat[:,item].A>0, dataMat[:,j].A>0))[0]
		#.A返回自身数据的2维数组的一个视图
		#logical_and 逻辑与，nonzero[0]返回非0元素的行坐标下标
		#找其他列向量，其中两个列向量都不为0的用户下标

		if len(overLap) == 0: 
			similarity = 0
		else:
			similarity = simMeas(dataMat[overLap,item], dataMat[overLap,j])
						#使用求相似度的方法
		#print "The %d and %d similarity is: %f" %(item, j, similarity)
		simTotal += similarity
		ratSimTotal += similarity * userRating
	if simTotal == 0:
		return 0
	else:
		return ratSimTotal/simTotal #归一化


#计算几维数据可以包含总能量的90%
def calEnergy(dataMat):
	U, sigma, VT = la.svd( dataMat )
	sig2 = sigma**2
	energy = sum(sig2) * 0.9 #得到总能量的90%
	#print energy
	for i in range( shape(sigma)[0]-1 ):
		#print sum(sig2[:(i+1)])
		if energy <= sum(sig2[:(i+1)]): break #如果大于90%
	sig_before = mat(eye(i+1) * sigma[:i+1])
	#print i+1, sum(sig2[:(i+1)])
	#print sig_before
	return i+1, sig_before, U

def svdEst(dataMat, user, item, simMeas):
	n = shape(dataMat)[1] #列数，即物品数量
	simTotal = 0.0
	ratSimTotal = 0.0
	num, sig_before, U = calEnergy(dataMat) #计算保留几个特征值
	#print num,shape(dataMat.T),shape(U[:,:num]),shape(sig_before.I)
	xformedItems = dataMat.T * U[:,:num] * sig_before.I  #构建转换后的物品
	print xformedItems
	for j in range(n):
		userRating = dataMat[user,j]
		if userRating == 0 or j==item: continue #评分为0或是其本身，跳过
		similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
		#print 'the %d and %d similarity is: %f' % (item, j, similarity)
		simTotal += similarity
		ratSimTotal += similarity * userRating
	if simTotal == 0: return 0
	else: return ratSimTotal/simTotal


#推荐引擎
def recommend(dataMat, user, N=99, simMeas=cosSim, estMethod=standEst):
	unratedItems = nonzero(dataMat[user,:].A==0)[1] #[1]代表返回列坐标
	#找出0元素的列下标，即找到未评分的物品
	if len(unratedItems) == 0:
		return "所有物品都已评分"
	itemScores = [] #初始化一个列表
	for item in unratedItems: #对未评分的物品进行遍历
		estimatedScore = estMethod(dataMat, user, item, simMeas)
		itemScores.append((item, estimatedScore)) #将预测的评分加入列表中
	return sorted(itemScores, key=lambda xx: xx[1], reverse=True)[:N]
	#reverse=True从大到小排列
	#key[1] 按estimatedScore排序
	#[:N] 返回前N组


if __name__=="__main__":
	#########对用户2,即矩阵第3行，进行评分预测
	user = 2
	dataMat = mat( loadExData() )
	#不同的距离算法
	#返回数据格式：[(物品，预测评分)]
	print "cosSim:", recommend(dataMat, user)
	print "eulidSim:", recommend(dataMat, user, simMeas=eulidSim)
	print "pearsSim:", recommend(dataMat, user, simMeas=pearsSim)
	##########################################

	#########利用SVD提高推荐的效果
	print "基于SVD的推荐"
	user = 2
	dataMat = mat( loadExData2() )
	print "cosSim:", recommend(dataMat, user, simMeas=cosSim, estMethod=svdEst)
	print "eulidSim:", recommend(dataMat, user, simMeas=eulidSim, estMethod=svdEst)
	print "pearsSim:", recommend(dataMat, user, simMeas=pearsSim, estMethod=svdEst)

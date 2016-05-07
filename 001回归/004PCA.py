#coding=utf-8
from numpy import *


def loadData(fileName):
	datMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = map(float, line.strip().split(' '))#strip()去掉\n
		datMat.append( curLine )
	fr.close()
	return mat(datMat)

def replaceNaN(datMat): #将NaN替换为当前列的平均值
	numFeat = shape(datMat)[1] #数据的特征数590
	#print numFeat
	for i in range(numFeat):
		tmp_no_isnan = ~isnan(datMat[:,i].A) #true或false的布尔数组
                        #.A生成数组而不是矩阵
		tmp_nonzero = nonzero(tmp_no_isnan)[0] #nonzero得到布尔数组的true的下标
		meanVal = mean( datMat[tmp_nonzero, i] ) #求非NaN的其他数的均值	
		datMat[nonzero(isnan(datMat[:,i].A))[0], i] = meanVal #NaN改为均值
	return datMat

def pca(datMat, N):
	meanVal = mean(datMat, axis=0) #求均值(按列相加求均值)
	datMat_meanRemove = datMat - meanVal #去除均值
	print "meanRemove datMat:",shape(datMat_meanRemove)
	covMat = cov(datMat_meanRemove, rowvar=0) #按每一列为一个变量
	eigVal, eigVect = linalg.eig(mat(covMat)) #求特征值和特征向量
	#得到行向量数组（特征值）和二维数组（每一列是一个特征向量）
	print "eigVal:", shape(eigVal)
	print "eigVect:", shape(eigVect)
	eigValInd = argsort(eigVal) #返回数组值从小到大排列的索引值,坐标从0开始
	eigValInd = eigValInd[:-(N+1):-1] #****重要××× 倒序取倒1至倒N
	print eigValInd
	redEigVect = eigVect[:, eigValInd] #按选定的特征值，选特征向量，一列为特征向量
	print "redEigVect:", shape(redEigVect)
	lowDataMat = datMat_meanRemove * redEigVect #将数据变换到新的维度
	print "lowDataMat:", shape(lowDataMat)
	#降维关键：原数据（取均值）乘以特征值
	reconMat = (lowDataMat * redEigVect.T) + meanVal #重构原始数据，用于调试
	return lowDataMat, reconMat

if __name__=="__main__":
	datMat = loadData("data004.data")
	datMat = replaceNaN(datMat)
	lowDataMat, reconMat = pca(datMat, 5)
	print shape(lowDataMat)





    

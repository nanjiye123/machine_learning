# -*- coding: utf-8 -*-
from numpy import *
import pylab as pl
#本程序是SVM的高斯内核版本

def kernelTrans(X, A, kTup): 
	#核函数将数据转换到高维空间，将2维转换到m维
	#kTup是一个元组，第一个参数给出核函数类型的一个字符串
	#X是所有数据点，A是一个数据点1×2
	m,n = shape(X)
	K = mat( zeros((m,1)) ) #K m行1列 列向量
	#=================================================
	if kTup[0] == 'lin': 
		K = X * A.T   #线性核函数
	elif kTup[0] == 'rbf':
		for j in range(m):                #遍历所有数据点
			deltaRow = X[j,:] - A         #xj-xi 1*2
			K[j] = deltaRow * deltaRow.T  #(1*2)(2*1)=(1*1)是一个数
		K = exp( K / (-1 * kTup[1]**2) )  #K是一个列向量，除以一个数
		#numpy的除法是数相除
	else: 
		raise NameError('核函数类型输入有误')
	#=================================================
	return K #K m*1 列向量

class optStruct: #高斯核版本
	def __init__(self, dataMatIn, classLabels, C, toler, kTup):
		self.X = dataMatIn                      #点的坐标
		self.labelMat = classLabels             #标签
		self.C = C                              #惩罚因子,不同优化问题的权重
		self.tol = toler                        #容错率，在innerL函数中，用于判断alphai
		self.m = shape(dataMatIn)[0]            #数据个数
		self.alphas = mat( zeros((self.m, 1)) ) #拉格朗日乘子
		self.b = 0                              #公式中的常数项
		self.eCache = mat( zeros((self.m, 2)) ) #存储误差值Ei和Ej
		#第一列是eCache是否有效的标志位，第二列是实际的Ek值
		self.K = mat( zeros((self.m, self.m)) ) #K是m*m的矩阵
		for i in range(self.m):	
			self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
			#核函数将2维转换到m维
			#K的每一列是转换后的一个数据点

def clipAlpha(aj,H,L):#用于调整alpha值，使其范围是H到L
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

def calcEk(oS, k): #误差值计算
	fxk = float( multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
	#alphas 100*1, labelMat 100*1, X 100*2 
	#print shape(fxk)
	Ek = fxk - float(oS.labelMat[k])
	#print shape(Ek)
	return Ek

def selectJrand(i,m):  #随机选择alphaj，i是第一个alpha下标
    j=i                #初始化第二个alpha下标
    while (j==i):      #随机生成第二个alpha下标，不等于第一个下标值
        j = int(random.uniform(0,m))#0到m生成一个实数
    return j
        
def selectJ(i, oS, Ei):   #启发式算法选择第二个alpha和计算Ej
	maxK = -1; maxDeltaE = 0; Ej = 0
	oS.eCache[i] = [1,Ei]  #将Ei在缓存中设置为有效（即已计算完成）  #choose the alpha that gives the maximum delta E
	validEcacheList = nonzero(oS.eCache[:,0].A)[0]#返回eCache第0列非零值的坐标，只要行号
	if (len(validEcacheList)) > 1:  #不是第一次循环
		for k in validEcacheList:   #循环所有有效的EcaChe值，找到差距最大的Ei-Ek，即最大步长
			if k == i: continue 	#k=i时，不用计算i，浪费时间
			Ek = calcEk(oS, k)
			deltaE = abs(Ei - Ek)   #计算差值
			if (deltaE > maxDeltaE):#选择具有最大步长的j
				maxK = k; maxDeltaE = deltaE; Ej = Ek
		return maxK, Ej
	else: #第一次循环，没有有效的eCache值
		j = selectJrand(i, oS.m) #随机选择一个alpha
		Ej = calcEk(oS, j)
	return j, Ej

def updateEk(oS, k): #对任意alpha值优化之后，计算误差值并存入缓存
	Ek = calcEk(oS, k)
	oS.eCache[k] = [1,Ek]
        
def innerL(i, oS):
	Ei = calcEk(oS, i) #用第一个alpha计算误差Ei
	#如果误差很大，需要对alpha进行优化，判断误差大小的条件如下
	if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
	#判断正间隔或负间隔<tol容错率 且 判断alpha是否在0到C内，而不是等于0或C。因为等于0或C就是在边界，不用优化。
		j,Ej = selectJ(i, oS, Ei)         #第二个alpha选择中的启发式算法，不是随机选择
		#==================================================
		#python通过引用传递列表，所以必须为alphaIold和alphaJold分配内存
		alphaIold = oS.alphas[i].copy()   #复制第一个alpha的值，作为老的alphai
		alphaJold = oS.alphas[j].copy()   #复制第二个alpha的值，作为老的alphaj
		#===================================================
		if (oS.labelMat[i] != oS.labelMat[j]):       #判断label同号还是异号
			L = max(0, oS.alphas[j] - oS.alphas[i])  #计算alpha2的上下限
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else: #label异号
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
		#===================================================
		if L==H:   #如果L=H，不更新两个alpha，直接退出
			print "L==H"
			return 0
		#===================================================
		#eta是alphaj的最优修改量，大于等于0时不更新，退出
		eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
		if eta >= 0: 
			print "eta>=0"
			return 0
		#===================================================
		oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta #更新alphaj
		oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)   #调整alphaj的值，使其在L到H的范围
		updateEk(oS, j)                              #更新alphaj误差值Ecache
		if (abs(oS.alphas[j] - alphaJold) < 0.00001):#alphaj变化得太小，不更新 
			print "j not moving enough"; return 0
		#===================================================
		#同时也改变alphai，两者改变的大小相同，但改变的方向相反，一个增加一个减少
		oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#更新alphai
		updateEk(oS, i) #更新alphai的误差值Ecache
		#===================================================
		#求b
		b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,i] - \
			 oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i,j]
		b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,j] - \
			 oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j,j]
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): 
			oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): 
			oS.b = b2
		else: 
			oS.b = (b1 + b2)/2.0
		#===================================================
		return 1
	else: return 0



#完整版Platt SMO外循环代码
#函数输入：数据集，类别标签，惩罚因子，容错率，最大循环次数
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup):
	oS = optStruct(mat(dataMatIn), mat(classLabels).T, C, toler, kTup)#构建数据结构，存储所有数据
	iter = 0                                      #迭代次数
	entireSet = True                              #标志整个数据集
	alphaPairsChanged = 0                         #记录优化的alpha个数
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
	#while终止条件：迭代次数超过最大值，上一次遍历没有对alpha进行修改 且 上一次遍历了所有alpha
		alphaPairsChanged = 0
		#=======================================================遍历整个数据集还是遍历非边界
		if entireSet:                             #遍历整个数据集
			for i in range(oS.m):                 #遍历所有数据点，遍历的是第一个alpha       
				alphaPairsChanged += innerL(i,oS) #调用innerL选择第二个alpha，有任意一对alpha改变返回1,否则返回0
				#print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
			iter += 1                             #迭代次数加1
		else:                                     #遍历所有的非边界alpha值，即不在边界0或C上的值
			nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]#取出值在0到C之间的alpha
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i,oS)
				#print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
			iter += 1                             #迭代次数加1
		#什么时候遍历整个数据集的alpha？
		#  上一次未遍历整个alpha且遍历的alpha无改变
		#什么时候遍历0到C之间的alpha？
		#  上一次的alpha有变化，不管是否遍历全部
		#=======================================================修改entireSet的真假值
		if entireSet: 
			entireSet = False
		elif (alphaPairsChanged == 0): 
			entireSet = True  
		print "iteration number: %d, %d" % (iter, alphaPairsChanged)
	return oS.b,oS.alphas


def loadData(fileName):
	datMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')#去掉回车和制表符
		datMat.append([ float(curLine[0]), float(curLine[1]) ])
		labelMat.append( float(curLine[2]) )#append是列表的方法
	fr.close()
	return datMat, labelMat

def calW(alpha, dataArr, labelArr):#计算w向量
	X = mat(dataArr)
	labelMat = mat(labelArr).T
	m, n = shape(X)
	w = zeros((n,1))
	print shape(alpha)
	print shape(labelMat)
	for i in range(m):#遍历所有样例
		w += multiply(alpha[i] * labelMat[i], X[i,:].T)
	return w

def drawpict(dataArr, labelArr, w = 0, b = 0):
#w维数2*1  b维数1*1
	x = mat(dataArr)
	for i in range(shape(labelArr)[0]):
		if labelArr[i]==1:
			pl.plot(x[i,0], x[i,1], 'ro')
		else:
			pl.plot(x[i,0], x[i,1], 'bs')
	pl.xlabel('x1')
	pl.ylabel('x2')
	pl.show()

def testRbf(k1 = 1.3): #参数k1是核函数中的到达率/函数值跌落速度
	#k1对结果的影响：
	#k1 = 0.1 时，核函数减少了每个支持向量的影响程度，因此需要更多的支持向量，支持向量的数目增加
	#k1 = 1.3 时，支持向量数目减少，且这些支持向量在决策边界的周围，此时测试错误率也下降
	#降低k1，训练错误率会降低，测试错误率会上升
	#===================================================================
	#   使用训练数据集
	dataArr,labelArr = loadData('data007.txt') #返回的是列表
	b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
	#惩罚因子C=200(重要)，容错率toler=0.0001，最大循环次数10000，核函数类型rbf
	datMat=mat(dataArr)                   #datMat 100*2
	labelMat = mat(labelArr).transpose()  #labelMat 100*1
	svInd=nonzero(alphas.A>0)[0]          #返回大于0的alpha的行号，即支持向量
	sVs=datMat[svInd]                     #得到支持向量的数据点
	labelSV = labelMat[svInd];            #得到支持向量的标签
	print "there are %d Support Vectors" % shape(sVs)[0]
	m,n = shape(datMat)
	errorCount = 0
	#-------------------------------------------------------------
	for i in range(m):#利用核函数进行分类
	#只需要支持向量数据就可以进行分类
		kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))      #求出类似于w的向量
		predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b #求出预测标签值
		if sign(predict)!=sign(labelArr[i]): errorCount += 1
	print "the training error rate is: %f" % (float(errorCount)/m)
	#-------------------------------------------------------------
	#====================================================================
	#    使用测试数据集
	dataArr,labelArr = loadData('data007_2.txt')
	errorCount = 0
	datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
	m,n = shape(datMat)
	#-------------------------------------------------------------
	for i in range(m):#利用核函数进行分类
	#只需要支持向量数据就可以进行分类
		kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
		predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
		if sign(predict)!=sign(labelArr[i]): errorCount += 1    
	print "the test error rate is: %f" % (float(errorCount)/m)  
	#-------------------------------------------------------------
	#====================================================================

if  __name__ == "__main__":
	testRbf(1.3)
	dataArr, labelArr = loadData("data007.txt")#返回的是列表
	drawpict(dataArr, labelArr)
	dataArr, labelArr = loadData("data007_2.txt")
	drawpict(dataArr, labelArr)


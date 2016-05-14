# -*- coding: utf-8 -*-
import pylab as pl
from collections import defaultdict,Counter


if __name__ == "__main__":

	#读取所有的数据点
	points = [[float(eachpoint.split("\t")[0]), float(eachpoint.split("\t")[1])] for eachpoint in open("data009.txt","r")]

	#计算每个数据点相邻的数据点，邻域定义为以该点为中心以边长为2*EPs的正方形网格
	Eps = 1
	surroundPoints = defaultdict(list) #创建类似字典的对象
	for idx1,point1 in enumerate(points): #enumerate遍历序列中的元素和下标 idx1下标，point1元素
		for idx2,point2 in enumerate(points):
			if (idx1 < idx2):
				if(abs(point1[0]-point2[0])<=Eps and abs(point1[1]-point2[1])<=Eps):
					surroundPoints[idx1].append(idx2) #存放下标
					surroundPoints[idx2].append(idx1)

	#定义邻域内相邻的数据点的个数大于等于MinPts为核心点
	MinPts = 7
	corePointIdx = [pointIdx for pointIdx,surPointIdxs in surroundPoints.iteritems() if len(surPointIdxs)>=MinPts]
	#corePointIdx是核心点集合

	#一个非核心点，其邻域内包含某个核心点，则定义这个非核心点为边界点
	borderPointIdx = [] #边界点集合
	for pointIdx,surPointIdxs in surroundPoints.iteritems(): #遍历邻域网络中的所有点
		if (pointIdx not in corePointIdx): #非核心点
			for onesurPointIdx in surPointIdxs:
				if onesurPointIdx in corePointIdx: #这个非核心点的邻域内有核心点
					borderPointIdx.append(pointIdx)
					break

	#噪音点既不是边界点也不是核心点
	#噪声点集合
	noisePointIdx = [pointIdx for pointIdx in range(len(points)) if pointIdx not in corePointIdx and pointIdx not in borderPointIdx]

	corePoint = [points[pointIdx] for pointIdx in corePointIdx]	    #核心点
	borderPoint = [points[pointIdx] for pointIdx in borderPointIdx] #边界点
	noisePoint = [points[pointIdx] for pointIdx in noisePointIdx]   #噪声点

	pl.subplot(121)
	pl.plot([eachpoint[0] for eachpoint in corePoint], [eachpoint[1] for eachpoint in corePoint], 'or')     #核心点
	pl.plot([eachpoint[0] for eachpoint in borderPoint], [eachpoint[1] for eachpoint in borderPoint], 'oy') #边界点
	pl.plot([eachpoint[0] for eachpoint in noisePoint], [eachpoint[1] for eachpoint in noisePoint], 'ok')   #噪声点

	#=============================================================================
	groups = [idx for idx in range(len(points))] #所有点作为一个簇，初始化
	
	#各个核心点与其邻域内的所有核心点放在同一个簇中
	for pointidx,surroundIdxs in surroundPoints.iteritems():
		for oneSurroundIdx in surroundIdxs:
			if (pointidx in corePointIdx and oneSurroundIdx in corePointIdx and pointidx < oneSurroundIdx):
				for idx in range(len(groups)):
					if groups[idx] == groups[oneSurroundIdx]:
						groups[idx] = groups[pointidx] #将该核心点放入邻域内核心点的簇中

	#边界点跟其邻域内的某个核心点放在同一个簇中
	for pointidx,surroundIdxs in surroundPoints.iteritems():
		for oneSurroundIdx in surroundIdxs:
			if (pointidx in borderPointIdx and oneSurroundIdx in corePointIdx):
				groups[pointidx] = groups[oneSurroundIdx]
				break

	#取簇规模最大的wantGroupNum个簇
	wantGroupNum = 4
	finalGroup = Counter(groups).most_common(wantGroupNum) #创建计数容器
	finalGroup = [onecount[0] for onecount in finalGroup]

	group1 = [points[idx] for idx in xrange(len(points)) if groups[idx]==finalGroup[0]]
	group2 = [points[idx] for idx in xrange(len(points)) if groups[idx]==finalGroup[1]]
	group3 = [points[idx] for idx in xrange(len(points)) if groups[idx]==finalGroup[2]]
	group4 = [points[idx] for idx in xrange(len(points)) if groups[idx]==finalGroup[3]]
	
	pl.subplot(122)
	pl.plot([eachpoint[0] for eachpoint in group1], [eachpoint[1] for eachpoint in group1], 'or')
	pl.plot([eachpoint[0] for eachpoint in group2], [eachpoint[1] for eachpoint in group2], 'oy')
	pl.plot([eachpoint[0] for eachpoint in group3], [eachpoint[1] for eachpoint in group3], 'og')
	pl.plot([eachpoint[0] for eachpoint in group4], [eachpoint[1] for eachpoint in group4], 'ob')

	#打印噪音点，黑色
	pl.plot([eachpoint[0] for eachpoint in noisePoint], [eachpoint[1] for eachpoint in noisePoint], 'ok')	

	pl.show()



#coding=utf-8
from numpy import *

def loadData(fileName):
    xMat = []
    yMat = []
    fr = open(fileName)
    for line in fr.readlines():
        tmp = []
        curLine = map(float, line.strip().split(','))#strip()去掉\n
        tmp.append( float(curLine[0]) )
        tmp.append( float(curLine[1]) )
        xMat.append( tmp )
        yMat.append( float(curLine[2]) )
    fr.close()
    return xMat, yMat

def add(m, xArr):
    xMat = mat(xArr).T
    tmp = m + xMat
    return tmp

def calc_sw(xArr, yArr, sw, m1, m2, total):
    xMat = mat(xArr).T
    tmp1 = sw
    tmp2 = sw
    for i in range(total):
        if yArr[i] == 1:
            tmp1 += dot( (xMat[:,i] - m1), (xMat[:,i] - m1).T )
        elif yArr[i] == 2:
            tmp2 += dot( (xMat[:,i] - m2), (xMat[:,i] - m2).T )
    #print tmp1, tmp2
    sw = tmp1 + tmp2
    return mat(sw)


def lda(xArr, yArr):
    n = len(xArr[0]) #确定维数
    total = len(yArr) #样本总数
    m1 = mat([0 for x in range(n)]).T #第一类的均值向量
    m2 = mat([0 for x in range(n)]).T #第二类的均值向量
    m = mat([0 for x in range(n)]).T
    num1 = 0 #第一类个数
    num2 = 0 #第二类个数
    for i in range(total):
        if yArr[i] == 1:
            m1 = add(m1, xArr[i])
            num1 += 1
        elif yArr[i] == 2:
            m2 = add(m2, xArr[i])
            num2 += 1
    m1 = divide(m1, num1) #m1是n*1维的向量
    m2 = divide(m2, num2)
    #print m1, m2

    sw = [ [] for x in range(n)]
    for i in range(n):
        sw[i] = [ 0 for x in range(n)]
    sw = calc_sw(xArr, yArr, sw, m1, m2, total)
    print "Sw矩阵：", sw
    inverse = linalg.inv(sw)
    print "Sw逆", inverse
    m = m2 - m1
    m = dot(inverse, m)
    print "w:", m
    return m

def cal_y(w, xArr):
    xMat = mat(xArr).T
    y = dot(w.T, xMat)
    print y
    return y


if __name__=="__main__":
    #数据格式声明 “1维，2维，分类1或2”
    xArr, yArr = loadData("data003.txt")
    #print xArr, yArr
    w = lda(xArr, yArr)
    cal_y(w, xArr)
    
    
    

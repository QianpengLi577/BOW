import numpy as np

def PCA(feature,k): #feature表示为输入数据   k表示需要的主成分  s.t. K≤特征维数
    m,n=feature.shape #size赋值
    ave=np.mean(feature,axis=0)  #获取均值
    b=np.cov(feature-ave,rowvar=False)  #获取协方差
    c,d=np.linalg.eig(b)  #求特征值、特征向量
    index=np.argwhere(np.imag(c)==0)  #获取特征值虚部为零的index
    eigen_true_value=np.real(c.ravel()[index])  #获取index对应的特征值
    eigen_vector=np.real(d[:,index])    #获得index对应的特征向量
    eigen_vector=np.reshape(eigen_vector,(-1,eigen_vector.shape[1]))  #特征向量reshape
    index1 = np.argsort(-eigen_true_value.T).T  #index进行排序
    if k>n  :
        print ("k≤特征维数")  #错误预警
        return
    else :
        selectVec = eigen_vector.T[index1[0:k]]  #获得前k个主成分向量
        selectVec=np.reshape(selectVec,(-1,selectVec.shape[2]))  #获得特征矩阵
        finalData = np.dot(feature-ave, selectVec.T)  #获得降维之后的数据
        #reconData = np.dot(finalData , selectVec) + ave
        return finalData,ave,selectVec

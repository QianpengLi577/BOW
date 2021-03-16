import numpy as np
import pandas as pd
from ksvd import ApproximateKSVD
#加载训练、测试数据
data_train=pd.read_excel('train_feature.xlsx').to_numpy()
data_test=pd.read_excel('test_feature.xlsx').to_numpy()
Xtr=data_train[:,0:128]
Xte=data_test[:,0:128]
Ytr=data_train[:,data_train.shape[1]-1]
Yte=data_test[:,data_test.shape[1]-1]
#归一化
from MKLpy.preprocessing import normalization, rescale_01
Xtr = rescale_01(Xtr)
Xtr = normalization(Xtr)

Xte = rescale_01(Xte)
Xte = normalization(Xte)

Xte=Xte.numpy()
Xtr=Xtr.numpy()
#KSVD稀疏编码
aksvd1 = ApproximateKSVD(n_components=350)
#字典1
dictionary1 = aksvd1.fit(Xtr).components_
#稀疏编码后的矩阵
X1 = aksvd1.transform(Xtr)

aksvd2 = ApproximateKSVD(n_components=600)
#字典二
dictionary2 = aksvd2.fit(X1).components_
#稀疏编码后的矩阵
X2 = aksvd2.transform(X1)
#获得test样本的表达矩阵
XTE=np.linalg.lstsq(np.dot(dictionary2,dictionary1).T, Xte.T, rcond=None)[0]
#线性SVM(OVA)
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X2, Ytr)
Ypr=np.array(clf.predict(XTE.T))

c=Ypr.shape[0]
#统计预测正确率
num=0
for i in range(c):
     if Ypr[i]==Yte[i]:
        num +=1
print ('Mult dictionary-ACC: %.3f' % (num*1.0/c))

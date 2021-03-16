import os
import joblib

print ('加载数据...', end='')
import pandas as pd
import numpy as np
#加载数据
data_train=pd.read_excel('train_feature_deep.xlsx').to_numpy()
data_test=pd.read_excel('test_feature_deep.xlsx').to_numpy()
#数据格式：行代表图像，列代表特征，最后一列为类别信息
Xtr=data_train[:,0:data_train.shape[1]-1]
Xte=data_test[:,0:data_test.shape[1]-1]
Ytr=data_train[:,data_train.shape[1]-1]
Yte=data_test[:,data_test.shape[1]-1]
print ('done')


print ('处理数据...', end='')
from MKLpy.preprocessing import normalization, rescale_01
#归一化
Xtr = rescale_01(Xtr)
Xtr = normalization(Xtr)

Xte = rescale_01(Xte)
Xte = normalization(Xte)
print ('done')

#线性SVM （OVA）
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(Xtr, Ytr)
Ypr=np.array(clf.predict(Xte))


c=Ypr.shape[0]
#统计预测正确率
num=0
for i in range(c):
     if Ypr[i]==Yte[i]:
        num +=1
print ('Deep feature-ACC: %.3f' % (num*1.0/c))


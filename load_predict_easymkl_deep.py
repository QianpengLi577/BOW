import torch
import joblib

print ('加载...', end='')
import pandas as pd
import numpy as np

#加载数据
data_train=pd.read_excel('train_feature_deep.xlsx').to_numpy()
data_test=pd.read_excel('test_feature_deep.xlsx').to_numpy()
Xtr=data_train[:,0:data_train.shape[1]-1]
Xte=data_test[:,0:data_test.shape[1]-1]
Ytr=data_train[:,data_train.shape[1]-1]
Yte=data_test[:,data_test.shape[1]-1]
print ('done')

#处理数据-归一化
print ('处理...', end='')
from MKLpy.preprocessing import normalization, rescale_01
Xtr = rescale_01(Xtr)
Xtr = normalization(Xtr)

Xte = rescale_01(Xte)
Xte = normalization(Xte)
print ('done')


Xtr=torch.as_tensor(Xtr)
Xte=torch.as_tensor(Xte)


print ('计算核矩阵...', end='')
num=12
#计算核矩阵
from MKLpy.metrics import pairwise
KLtr = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(num)]
KLte = [pairwise.homogeneous_polynomial_kernel(Xte,Xtr, degree=d) for d in range(num)]
print ('done')


#MKL
from MKLpy.algorithms import AverageMKL, EasyMKL
from sklearn.svm import SVC
base_learner = SVC(C=0.1)
clf = EasyMKL(lam=0.1, multiclass_strategy='ova', learner=base_learner).fit(KLtr,Ytr)
#预测
from sklearn.metrics import accuracy_score, roc_auc_score
y_pred = clf.predict(KLte)
y_score = clf.decision_function(KLte)
accuracy = accuracy_score(Yte, y_pred)
print ('Deep feature-ACC: %.3f' % (accuracy))

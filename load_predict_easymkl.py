import torch
import joblib

print ('加载中...', end='')
import pandas as pd
#加载之前保存的BOF数据  包括均值、类名、特征向量、单词数、特征单词
mean, classes_names, var, k, voc = joblib.load("bof.pkl")

#加载数据
data_train=pd.read_excel('train_feature.xlsx').to_numpy()
data_test=pd.read_excel('test_feature.xlsx').to_numpy()
Xtr=data_train[:,0:data_train.shape[1]-1]
Xte=data_test[:,0:data_test.shape[1]-1]
Ytr=data_train[:,data_train.shape[1]-1]
Yte=data_test[:,data_test.shape[1]-1]
print ('ok')

#处理数据归一化
print ('处理中...', end='')
from MKLpy.preprocessing import normalization, rescale_01
Xtr = rescale_01(Xtr)
Xtr = normalization(Xtr)

Xte = rescale_01(Xte)
Xte = normalization(Xte)
print ('ok')


Xtr=torch.as_tensor(Xtr)
Xte=torch.as_tensor(Xte)

#计算核
print ('计算核...', end='')
from MKLpy.metrics import pairwise
KLtr = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(5)]
KLte = [pairwise.homogeneous_polynomial_kernel(Xte,Xtr, degree=d) for d in range(5)]
print ('done')


#MKL
from MKLpy.algorithms import EasyMKL


print ('OVA EasyMKL...', end='')
from sklearn.svm import SVC
base_learner = SVC(C=0.1)
clf = EasyMKL(lam=0.1, multiclass_strategy='ova', learner=base_learner).fit(KLtr,Ytr)
print('done')

from sklearn.metrics import accuracy_score, roc_auc_score
y_pred = clf.predict(KLte)
y_score = clf.decision_function(KLte)
accuracy = accuracy_score(Yte, y_pred)
print ('SIFT+HOG+EasyMKL-ACC: %.3f' % (accuracy))

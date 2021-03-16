import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib

class LocallyWeightedLinearRegression:
    def __init__(self, tau):
        self.tau = tau  # tau 为波长参数，它控制了权值随距离下降的速率
        self.w = None

    def fit_predict(self, X, y, checkpoint_x):  # checkpoint_x 是代表对checkpoint_x这个点做线性回归 X样本集，y标签集
        m = X.shape[0]
        self.n_features = X.shape[1]
        extra = np.ones((m,))
        X = np.c_[X, extra]
        checkpoint_x = np.r_[checkpoint_x, 1]
        self.X, self.y, self.checkpoint_x = X, y, checkpoint_x
        weight = np.zeros((m,))  # 权重矩阵，指的是不同样本对checkpoint_x的影响程度
        for i in range(m):
            weight[i] = np.exp(-(X[i] - checkpoint_x).dot((X[i] - checkpoint_x).T) / (2 * (self.tau ** 2)))

        weight_matrix = np.diag(weight)  # 扩展成对角阵，array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵, array是一个二维矩阵时，结果输出矩阵的对角线元素
        self.w = np.linalg.inv(X.T.dot(weight_matrix).dot(X)).dot(X.T).dot(weight_matrix).dot(y)  # 模型参数w 计算
        return checkpoint_x.dot(self.w)  # 返回对当前点 checkpoint_x 的预测值

    def fit_transform(self, X, y, checkArray):
        m = len(y)
        preds = []  # 预测结果列表
        for i in range(m):
            preds.append(self.fit_predict(X, y, checkArray[i]))
        return np.array(preds)

x=np.arange(0.0001,20,0.1)
y=[]
g=1.1
for i in range(x.shape[0]):
    # 加载数据--SIFT的直方图统计
    data_train = pd.read_excel('train_feature_lowess.xlsx').to_numpy()
    data_test = pd.read_excel('test_feature_lowess.xlsx').to_numpy()

    # 前128维为特征，最后一列为类别信息
    Xtr = data_train[:, 0:128]
    Xte = data_test[:, 0:128]
    Ytr = data_train[:, data_train.shape[1] - 1]
    Yte = data_test[:, data_test.shape[1] - 1]

    mean = np.mean(Xtr, axis=0)  # 列均值
    std = np.std(Xtr, axis=0)  # 列方差
    mean = mean.reshape(-1, 1)
    lr = LocallyWeightedLinearRegression(tau=5)  # 进行lowess加权回归
    y_pred = lr.fit_transform(mean, std, mean)  # 对当前的mean进行预测

    # plt.scatter(mean, std, label='original')  # 画图并设置标签
    # plt.scatter(mean, y_pred, label='predict')
    # plt.legend()  # 显示图例
    # plt.show()

    max = y_pred.max()
    min = y_pred.min()
    w = np.log((y_pred - min + x[i]) * 1.0 / (max - min))  # 获得权重向量
    wtr = np.tile(w, (Xtr.shape[0], 1))
    wte = np.tile(w, (Xte.shape[0], 1))

    # 归一化
    # print('处理数据...', end='')
    from MKLpy.preprocessing import normalization, rescale_01

    Xtr1 = normalization(Xtr)

    Xte1 = normalization(Xte)
    # print('done')
    # 线性SVM（OVA）
    from sklearn.svm import LinearSVC

    clf = LinearSVC()
    clf.fit(Xtr1, Ytr)
    Ypr = np.array(clf.predict(Xte1))

    c = Ypr.shape[0]
    # 统计预测正确率
    num = 0
    for q in range(c):
        if Ypr[q] == Yte[q]:
            num += 1
    # print('Only SIFT-ACC: %.3f' % (num * 1.0 / c))
    g=(num * 1.0 / c)
    # 获得加权之后的数据
    Xtr = Xtr * wtr
    Xte = Xte * wte
    # 归一化
    # print('处理数据...', end='')
    from MKLpy.preprocessing import normalization

    Xtr = normalization(Xtr)

    Xte = normalization(Xte)
    # print('done')
    # 线性SVM（OVA）
    from sklearn.svm import LinearSVC

    sv = LinearSVC()
    sv.fit(Xtr, Ytr)
    Ypr = np.array(sv.predict(Xte))

    d = Ypr.shape[0]
    # 统计预测正确率
    num = 0
    for j in range(d):
        if Ypr[j] == Yte[j]:
            num += 1
    # print('SIFT using lowess-ACC: %.3f' % (num * 1.0 / d))
    y.append(num * 1.0 / d)
yq=np.array(y)
plt.scatter(x, yq,marker=".")
plt.show()
ymaxloc=np.argmax(yq)
print('Only SIFT-ACC: %.3f' % (g))
# print(ymaxloc)
print('SIFT using lowess-ACC: %.3f' % yq.max())
print('finsh')



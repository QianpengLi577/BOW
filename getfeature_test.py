#该文件与getfeature_train文件的代码基本相同，故仅简略注释
import cv2
import imutils
import numpy as np
import os
import pandas as pd
from skimage import feature
import sys
from scipy.cluster.vq import *
np.set_printoptions(threshold=sys.maxsize)
import joblib
from sklearn.preprocessing import StandardScaler


# 加载model
mean, classes_names, var, k, voc = joblib.load("bof.pkl")

# 获得训练集
def mylistdir(directory):
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]
#SIFT特征
sift = cv2.xfeatures2d.SIFT_create()

test_path = "data20/test1/"  # 文件路径
testing_names = mylistdir(test_path)  # 每一类的名字

image_paths = []
image_classes = []
class_id = 0
#获取图像url、类别信息
for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = imutils.imlist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

des_list = []
d_list = []

for image_path in image_paths:
    #获取SIFT特征
    im = cv2.imread(image_path)
    kpts, des = sift.detectAndCompute(im, None)
    des_list.append((image_path, des))
    #获取HOG特征
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    fd = feature.hog(img, orientations=9, pixels_per_cell=(16, 16),
                     cells_per_block=(4, 4), block_norm='L1',
                     visualize=False, transform_sqrt=True)
    #对HOG特征进行放大处理
    v1 = np.array(fd)
    v1=v1*10000
    d_list.append((image_path, v1))
#sift特征存储
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))
#HOG特征存储
des = d_list[0][1]
for image_path, descriptor in d_list[1:]:
    des = np.vstack((des, descriptor))
#初始化图像直方图矩阵
test_features = np.zeros((len(image_paths), k), "float32")
#对图像进行VQ编码
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1
#获取PCA降维之后的HOG特征
q=np.dot(des-mean, var.T)
#类别转置便于写入文件
classes=np.array([image_classes]).T
df1 = pd.DataFrame(np.c_[test_features,np.c_[q,classes]])
df1.to_excel('test_feature_lowess.xlsx', sheet_name='train_feature', index=False)
#标准化并写入文件
scaler = StandardScaler()
q = scaler.fit_transform(q)


test_features = scaler.fit_transform(test_features)
#sift与HOG级联
df = pd.DataFrame(np.c_[test_features,np.c_[q,classes]])
df.to_excel('test_feature.xlsx', sheet_name='test_feature', index=False)



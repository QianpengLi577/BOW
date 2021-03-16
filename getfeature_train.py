import cv2
import imutils
import numpy as np
import os
import pandas as pd
from skimage import feature
import PCA
from scipy.cluster.vq import *
import joblib
from sklearn.preprocessing import StandardScaler

#获取类名的函数
def mylistdir(directory):
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

sift = cv2.xfeatures2d.SIFT_create()  #sift特征获取函数

train_path = "data20/train1/"  # 文件路径
training_names = mylistdir(train_path)  # 每一类的名字

image_paths = []   #存放每一张图片的位置
image_classes = []   #存放每一张图片的类别信息，从0开始
class_id = 0  #标记类别信息
for training_name in training_names:
    dir = os.path.join(train_path, training_name)  #当前类别的路径
    class_path = imutils.imlist(dir)    #当前类别所有图片的路径
    image_paths += class_path
    image_classes += [class_id] * len(class_path)   #图像类别生成
    class_id += 1  #当前类别存储完后进行下一个类别信息的存储、生成
# 存放描述符(sift)
des_list = []  #存储图片的sift特征
d_list = []  #存储图片的HOG特征

for image_path in image_paths:
    im = cv2.imread(image_path)   #获得img
    kpts, des = sift.detectAndCompute(im, None)   #sift检测
    des_list.append((image_path, des))   #将每一张图片的sift特征存储起来

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  #灰度化图像
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)  #将图像调整至128*128
    fd = feature.hog(img, orientations=9, pixels_per_cell=(16, 16),
                     cells_per_block=(4, 4), block_norm='L1',
                     visualize=False, transform_sqrt=True)    #获取HOG特征，3600维
    v1 = np.array(fd)   #list转化为array
    v1=v1*10000    #将特征向量乘以10000，便于后续的PCA降维
    d_list.append((image_path, v1))   #存储HOG特征


descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))
#将所有图像的sift特征进行堆叠
des = d_list[0][1]
for image_path, descriptor in d_list[1:]:
    des = np.vstack((des, descriptor))
#将所有图像的HOG特征进行堆叠

k = 128   #定义字典单词数维128
voc, variance = kmeans(descriptors, k, 1)   #Kmeans聚类，获得k个聚类中心


# 初始化直方图矩阵
im_features = np.zeros((len(image_paths), k), "float32")
#使用矢量量化的方法对图像进行编码
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

q,mean,var=PCA.PCA(des,128)  #进行PCA降维，降至128维
classes=np.array([image_classes]).T  #将类别信息进行转置，用于存储

#用于lowess处理的数据(不做归一化处理)
df1 = pd.DataFrame(np.c_[im_features,np.c_[q,classes]])
df1.to_excel('train_feature_lowess.xlsx', sheet_name='train_feature', index=False)
#将HOG特征进行归一化(均值为0，方差为1)
scaler = StandardScaler()
q = scaler.fit_transform(q)

#将SIFT特征进行归一化
im_features = scaler.fit_transform(im_features)
#保存SIFT、HOG、类别信息
df = pd.DataFrame(np.c_[im_features,np.c_[q,classes]])
df.to_excel('train_feature.xlsx', sheet_name='train_feature', index=False)

#保存model
joblib.dump((mean, training_names, var, k, voc), "bof.pkl", compress=3)

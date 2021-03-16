#该文件与getfeature_train_deep文件的代码基本相同，相同处简略注释，不同处详细注释
import imutils
import numpy as np
import os
import pandas as pd
import keras
#获取类文件url
def mylistdir(directory):
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

train_path = "data20/test1/"  # 文件路径
training_names = mylistdir(train_path)  # 每一类的名字

image_paths = []
image_classes = []
class_id = 0
##获取图像url、类别信息
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

des_list = []
d_list = []

from extract_cnn_vgg16_keras import VGGNet
#构建VGGmodel
model = VGGNet()
#对每一张图片获取深度特征--512维特征向量
for image_path in image_paths:
    des = model.extract_feat(image_path)
    des_list.append((image_path, des))
#整合特征向量
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))
#类别转置便于存储
classes=np.array([image_classes]).T
#写入文件
df = pd.DataFrame(np.c_[descriptors,np.c_[classes]])
df.to_excel('test_feature_deep.xlsx', sheet_name='train_feature', index=False)



import numpy as np
from numpy import linalg as LA
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

class VGGNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)  #输入图像尺寸为224*224*3
        self.pooling = 'max'      #池化方式为max pooling
        self.model = VGG16(weights = 'imagenet', 
                           input_shape = (self.input_shape[0], 
                           self.input_shape[1], self.input_shape[2]), pooling = self.pooling, include_top = False)  #VGG16网络参数设置
        self.model.predict(np.zeros((1, 224, 224 , 3)))  #predict初始化

    # 获取最后一层卷积输出的特征
    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))   #加载图像
        img = image.img_to_array(img)  #整数矩阵变浮点矩阵
        img = np.expand_dims(img, axis=0)  #升维，将原图像224*224*3升维到1*224*224*3
        img = preprocess_input(img)   #预处理
        feat = self.model.predict(img)  #提取图像特征
        norm_feat = feat[0]/LA.norm(feat[0])  #归一化特征
        return norm_feat


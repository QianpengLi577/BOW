import cv2
import numpy as np
import math

class Hog_descriptor():
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img  #img=img
        self.img = np.sqrt(img / np.max(img))  #gamma矫正--归一化到0~1
        self.img = img * 255  #放大到0~255
        self.cell_size = cell_size  #cell尺寸的初始化
        self.bin_size = bin_size  #bin的初始化
        self.angle_unit = 360 / self.bin_size  #计算一个bin对应的角度

    def extract(self):
        #图像size赋值
        height, width = self.img.shape
        # 计算图像的梯度大小和方向
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        #初始化每个cell的特征矩阵
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                # cell内的梯度大小
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                # cell内的梯度方向
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                # 转化为梯度直方图格式
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        # block组合、归一化
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                #block维2*2个cell
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                #计算当前block的元素平方和
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    #归一化
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector
    #图像的梯度大小、角度计算
    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)#x方向sobel算子
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)#y方向sobel算子
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)#线性叠加得到梯度大小
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)#计算梯度方向
        return gradient_magnitude, gradient_angle

    #计算cell直方图
    def cell_gradient(self, cell_magnitude, cell_angle):
        #初始化cell特征向量
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                #获得当前cell的梯度大小和梯度方向
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                #得到当前角度属于的bin、权数
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                #进行直方图统计
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    #获得当前角度属于的最小bin(向下取整)、最大bin、余数加权
    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) , mod

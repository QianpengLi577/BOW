from keras import utils
from keras.datasets import mnist
from keras.models import Model
from keras.layers import *
from keras import backend as K
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""    ###这两句代码用来关闭CUDA加速
from keras.models import Input,Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D,GlobalAveragePooling2D
from keras.layers import BatchNormalization,Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

batch_size = 12
num_classes = 0
img_rows, img_cols = 128, 128

def mylistdir(directory):
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

train_path = "E:/My_task/data20/train1/"  # 文件路径
training_names = mylistdir(train_path)  # 每一类的名字
test_path = "E:/My_task/data20/test1/"  # 文件路径
testing_names = mylistdir(test_path)  # 每一类的名字

def imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if (f.endswith('.jpg') or f.endswith('.png'))]

image_paths_test = []
image_classes_test = []
class_id_test = 0
image_paths_train = []
image_classes_train = []
class_id_train = 0
for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = imlist(dir)
    image_paths_test += class_path
    image_classes_test += [class_id_test] * len(class_path)
    class_id_test += 1
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imlist(dir)
    image_paths_train += class_path
    image_classes_train += [class_id_train] * len(class_path)
    class_id_train += 1
num_classes=class_id_train

img_list_test = []
for image_path in image_paths_test:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
    img_list_test.append(img)
imgtest=np.array(img_list_test)
img_list_train = []
for image_path in image_paths_train:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
    img_list_train.append(img)
imgtrain=np.array(img_list_train)

x_test = imgtest.reshape(len(image_classes_test), img_rows, img_cols, 1)
y_test=np.array(image_classes_test)
x_train = imgtrain.reshape(len(image_classes_train), img_rows, img_cols, 1)
y_train=np.array(image_classes_train)
y_test_label=y_test

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# CNNtrain
input_data = Input(shape=(None,None,1))
out = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(input_data)
out = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(out)
out = BatchNormalization(axis=3)(out)
out = MaxPool2D(pool_size=(2,2),strides=(2,2))(out)
out = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
out = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
out = BatchNormalization(axis=3)(out)
out = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(out)
out = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
out = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
out = BatchNormalization(axis=3)(out)
out = GlobalAveragePooling2D()(out)
####out = Dense(units=512,activation='relu')(out) ###数据集比较小，可不加这一层
out = Dense(units=6,activation='softmax')(out)
model = Model(inputs=input_data,outputs=out)
model.summary()
model_check = ModelCheckpoint('./model.h5',
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='auto',
                                  period=1)  ####用来验证每一epoch是否是最好的模型用来保存  val_loss
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=30,
          verbose=1,
          validation_split=0.3)
model.save('./CNN_model.h5')

#predict
from keras.models import load_model
model = load_model('./CNN_model.h5')
test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_acc)

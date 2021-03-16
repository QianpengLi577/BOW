import cv2, os

def imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if (f.endswith('.jpg') or f.endswith('.png'))]
#获得图片的url

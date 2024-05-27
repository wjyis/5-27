import os
import cv2 as cv

def read_data(path, method = "gray"):
    imgs = []
    color_method = 1 if method == "gray" else 0
    listdir = sorted(os.listdir(path))
    for i in listdir:
        imgs.append(cv.imread(f"{path}/{i}", color_method)/255)
    return imgs


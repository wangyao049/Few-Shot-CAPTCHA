import cv2
import numpy as np
import os
import random

cnnPath = r'./results/noise2denoise/test_latest/images/'
cnnCyclePath = r'../data/raw_test/CNN'

for fileName in os.listdir(cnnPath):
    if fileName == ".DS_Store":
        continue
    name = fileName.split(".")[0]
    flag = name.split("_")[1]
    imgs = []
    if flag == "fake":
        img = cv2.imread(cnnPath+fileName)
        # 不调整尺寸后面分割效果更好因此不进行尺寸调整
        newName = name.split("_")[0]
        if os.path.exists(cnnCyclePath) is False:
            os.makedirs(cnnCyclePath)
        FileSavePath = '{0}/{1}.jpg'.format(cnnCyclePath, newName)
        print("saving " + FileSavePath)
        cv2.imwrite(FileSavePath, img)






import os
import random
import cv2

filePath = r'./data/processed_train/'
trainsets = os.listdir(filePath)
if 'gen5' in trainsets:
    trainsets.remove('gen5')
print(trainsets)
savePath = r'./data/processed_train/gen5/'

all = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
       'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
       'v', 'w', 'x', 'y', 'z',
       'AA', 'BB', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'JJ', 'KK', 'LL', 'MM', 'NN', 'OO', 'PP', 'QQ', 'RR',
       'SS', 'TT', 'UU', 'VV', 'WW', 'XX', 'YY', 'ZZ']

# 五个数据集各选五张图像
for dataset in trainsets:
    if dataset == '.DS_Store':
        continue
    for ch in all:
        p = filePath + dataset + '/' + ch
        imgs = os.listdir(p)
        if '.DS_Store' in imgs:
            imgs.remove('.DS_Store')
        random.shuffle(imgs)
        for m in imgs[:5]:
            img = cv2.imread(p + '/' + m)
            save_dir = savePath + ch
            if os.path.exists(save_dir) is False:
                os.makedirs(save_dir)
            cv2.imwrite(save_dir + '/' + m, img)
import numpy as np
import os.path
from PIL import Image
import json
import random



all = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
       'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
       'v', 'w', 'x', 'y', 'z',
       'AA', 'BB', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'JJ', 'KK', 'LL', 'MM', 'NN', 'OO', 'PP', 'QQ', 'RR',
       'SS', 'TT', 'UU', 'VV', 'WW', 'XX', 'YY', 'ZZ']

def getLabels(path):
    train_labels = all.copy()
    labels = os.listdir(path+'/images/')
    test_labels = labels[:16]
    for l in test_labels:
        train_labels.remove(l)
    return train_labels, test_labels


def getDic(labels, dataPath, savePath, dic):
    i = 0
    for label in labels:
        dic['label_names'].append(label)
        imgs = os.listdir(dataPath + label + "/")
        random.shuffle(imgs)
        if len(imgs) >= 26:
            # print(len(imgs))
            newImgs = []
            for img in imgs:
                if img == '.DS_Store':
                    continue
                newImgs.append(img)
            # print(len(newImgs))
            for im in newImgs[:25]:
                image_path = savePath + label + "/" + im
                dic['image_names'].append(image_path)
                dic['image_labels'].append(i)
        else:
            I = []
            I.extend(imgs)
            imgs.extend(I)
            while len(imgs) < 26:
                imgs.extend(I)
            newImgs = []
            # print(len(imgs))
            for img in imgs:
                if img == '.DS_Store':
                    continue
                newImgs.append(img)
            # print(len(newImgs))
            random.shuffle(newImgs)
            for im in newImgs[:25]:
                image_path = savePath + label + "/" + im
                dic['image_names'].append(image_path)
                dic['image_labels'].append(i)
        i = i+1

def getMultiDic(labels, dataPath, savePath, dic):
    i = 0
    for label in labels:
        dic['label_names'].append(label)
        imgs = os.listdir(dataPath + label + "/")
        print(len(imgs))
        for im in imgs:
            if im == '.DS_Store':
                continue
            image_path = savePath + label + "/" + im
            dic['image_names'].append(image_path)
            dic['image_labels'].append(i)
        i = i+1
        # print(len(imgs))


# def getDics(labels, dataPath, savePath1, savePath2, dic1, dic2):
#     i = 0
#     for label in labels:
#         dic1['label_names'].append(label)
#         dic2['label_names'].append(label)
#         imgs = os.listdir(dataPath + label + "/")
#         random.shuffle(imgs)
#         if len(imgs) >= 26:
#             # print(len(imgs))
#             newImgs = []
#             for img in imgs:
#                 if img == '.DS_Store':
#                     continue
#                 newImgs.append(img)
#             # print(len(newImgs))
#             for im in newImgs[:25]:
#                 image_path1 = savePath1 + label + "/" + im
#                 dic1['image_names'].append(image_path1)
#                 dic1['image_labels'].append(i)
#                 image_path2 = savePath2 + label + "/" + im
#                 dic2['image_names'].append(image_path2)
#                 dic2['image_labels'].append(i)
#         else:
#             I = []
#             I.extend(imgs)
#             imgs.extend(I)
#             if len(imgs) < 26:
#                 imgs.extend(I)
#             newImgs = []
#             # print(len(imgs))
#             for img in imgs:
#                 if img == '.DS_Store':
#                     continue
#                 newImgs.append(img)
#             # print(len(newImgs))
#             random.shuffle(newImgs)
#             for im in newImgs[:25]:
#                 image_path1 = savePath1 + label + "/" + im
#                 dic1['image_names'].append(image_path1)
#                 dic1['image_labels'].append(i)
#                 image_path2 = savePath2 + label + "/" + im
#                 dic2['image_names'].append(image_path2)
#                 dic2['image_labels'].append(i)
#         i = i+1

def write_to_json(dic, fileName):
    with open(fileName, 'w', encoding='utf-8') as f:
        json.dump(dic, f, indent=4)


datasets = ['cnn','bjyh','wiki','gfyh','msyh']

def sort(a):
    return len(a)

filePath = r'./filelists/'


for ds in datasets:
    train_labels, test_labels = getLabels(filePath+ds)
    print(ds,":",test_labels)
    random.shuffle(test_labels)
    novel_labels = test_labels[:8]
    val_labels = test_labels[8:]

    # print(testset,":",test_labels)
    jsPath = filePath + ds + '/'
    basePath = filePath + 'captcha/'
    multiPath = filePath + 'gen5/'
    testPath = filePath + ds + '/' + 'images/'
    baseFileName = jsPath + 'base.json'
    novelFileName = jsPath + 'novel.json'
    valFileName = jsPath + 'val.json'
    multiFileName = jsPath + 'multiBase.json'

    savePath1 = '/root/wangyao/zyf/EXP/CRFSL/filelists/' + 'captcha/'
    savePath3 = '/root/wangyao/zyf/EXP/CRFSL/filelists/' + 'gen5/'
    savePath2 = '/root/wangyao/zyf/EXP/CRFSL/filelists/'+ds+'/images/'

    base_dic = {'label_names': [], 'image_names': [], 'image_labels': []}
    multi_dic = {'label_names': [], 'image_names': [], 'image_labels': []}
    novel_dic = {'label_names': [], 'image_names': [], 'image_labels': []}
    val_dic = {'label_names': [], 'image_names': [], 'image_labels': []}

    getDic(train_labels, basePath, savePath1, base_dic)
    write_to_json(base_dic,baseFileName)

    getMultiDic(train_labels, multiPath, savePath3, multi_dic)
    write_to_json(multi_dic, multiFileName)

    getDic(novel_labels, testPath, savePath2, novel_dic)
    write_to_json(novel_dic, novelFileName)

    getDic(val_labels, testPath, savePath2, val_dic)
    write_to_json(val_dic, valFileName)





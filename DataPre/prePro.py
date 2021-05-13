'''
opencv做图像处理，所以需要安装下面两个库
pip3 install opencv-python
'''
 
import cv2
import numpy as np
import os
import random
 
trainFilePath = r'data/raw_train/'  # 原始验证码路径
testFilePath = r'data/raw_test/'

savePath_cpt = r'data/processed_train/captcha'  # 切割后的captcha验证码的存放路径
savePath_gen1 = r'data/processed_train/processed_patchca1'
savePath_gen2 = r'data/processed_train/processed_patchca2'
savePath_gen3 = r'data/processed_train/processed_patchca3'
savePath_gen4 = r'data/processed_train/processed_patchca4'


savePath_wiki = r'data/processed_test/wiki/images'  # 切割后的真实验证码的存放路径
savePath_gfyh = r'data/processed_test/gfyh/images'
savePath_msyh = r'data/processed_test/msyh/images'
savePath_CNN = r'data/processed_test/cnn/images'
savePath_bjyh = r'data/processed_test/bjyh/images'

n = 0

trainPathList = os.listdir(trainFilePath)
print(trainPathList)
testPathList = os.listdir(testFilePath)
print(testPathList)
Big = "QWERTYUIOPASDFGHJKLZXCVBNM"

def getname(name):
    n = ""
    for i in name:
        n+=i
    return n

# 预处理+轮廓识别分割
# for path in trainPathList:
#     # python_captcha
#     if path == "captcha":
#         for fileName in os.listdir(trainFilePath + path + "/"):
#             if fileName == ".DS_Store":
#                 continue
#             name = fileName.split(".")[0]
#             newname = []
#             for n in name:
#                 if n in Big:
#                     n = n + n
#                 newname.append(n)
#             name = newname
#             filePathIm = trainFilePath + path + "/" + fileName
#             im1 = cv2.imread(filePathIm)  # 读入图片
#             im_gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)  # 将图片转成灰度图
#
#             # 报错 error: (-215:Assertion failed) 原因是文件路径错误
#             ret1, im_inv1 = cv2.threshold(im_gray1, 200, 255, cv2.THRESH_BINARY_INV)  # 二值化
#
#             # 应用高斯模糊对图片进行降噪，高斯模糊的本质是用高斯核和图像做卷积
#             kernel1 = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [2, 4, 2]])
#             im_blur1 = cv2.filter2D(im_inv1, -1, kernel1)
#             # 降噪后再做一轮二值化处理
#             ret2, im_inv2 = cv2.threshold(im_blur1, 200, 255, cv2.THRESH_BINARY)
#             # cv2.imshow("image", im_inv2)
#             # cv2.waitKey()
#             # 把最开始的图片切割成单个字符
#             # 第一步 用opencv的findContours来提取轮廓 （cv2.findContours()函数接受的参数为二值图）
#             contours, hierarchy = cv2.findContours(im_inv2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             # 第一个参数是寻找轮廓的图像，第二个参数表示轮廓的检索模式，第三个参数method为轮廓的近似办法
#             cv2.drawContours(im1, contours, -1, (0, 255, 0), 1)  # 第三个参数为线条颜色，第四个参数线条粗度
#
#             num = 0
#             result = []
#             for i in range(len(contours)):
#                 if 1400 > cv2.contourArea(contours[i]) > 140:
#                     num += 1
#                     # cv2.drawContours(cv2.imread(filepath1) , contours, i, (0, 255, 0), 1)   # i 表示绘制第i条轮廓
#                     x, y, w, h = cv2.boundingRect(contours[i])  # 用一个最小的矩形，把找到的形状包起来
#                     result.append([x, x + w, y, y + h])
#                     im3 = cv2.rectangle(im1, (x, y), (x + w, y + h), (0, 0, 255), 2)
#             if len(result) == 4:
#                 i = 0
#                 result.sort(key=(lambda x: x[0]))
#                 for [x, x1, y, y1] in result:
#                     roi = im_inv1[y:y1, x:x1]
#                     if os.path.exists('{0}/{1}'.format(savePath_cpt, name[i])) is False:
#                         os.makedirs('{0}/{1}'.format(savePath_cpt, name[i]))
#                     fileSavePath = '{0}/{1}/{2}.jpg'.format(savePath_cpt, name[i],
#                                                             path + "_" + getname(name) + "_" + name[i])
#                     print(fileSavePath)
#                     roi = cv2.resize(roi, (30, 30))
#                     cv2.imwrite(fileSavePath, roi)
#                     i += 1
#                 cv2.imshow("image", im3)
#                 cv2.waitKey(100)  # 100毫秒
#
#     # java_patchca
#     if path in ['patchca1', 'patchca2', 'patchca4', 'patchca3']:
#         for fileName in os.listdir(trainFilePath + path + "/"):
#             if fileName == ".DS_Store":
#                 continue
#             name = fileName.split(".")[0]
#             newname = []
#             for n in name:
#                 if n in Big:
#                     n = n + n
#                 newname.append(n)
#             name = newname
#             if fileName == ".DS_Store":
#                 continue
#             filePathIm = trainFilePath + path + "/" + fileName
#             img = cv2.imread(filePathIm)  # 读入图片
#             # print(filePathIm)
#             hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 将图片转成灰度图
#             lower = np.array([0, 0, 160])
#             upper = np.array([244, 65, 255])
#             mask = cv2.inRange(hsv, lower, upper)
#             ret1, im_inv1 = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY_INV)  # 二值化
#             kernel1 = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [2, 4, 2]])
#             im_blur1 = cv2.filter2D(im_inv1, -1, kernel1)
#             ret2, im_inv2 = cv2.threshold(im_blur1, 200, 255, cv2.THRESH_BINARY)
#             # cv2.imshow("image", im_inv2)
#             # cv2.waitKey(200)
#             contours, hierarchy = cv2.findContours(im_inv2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             # cv2.drawContours(img, contours, -1, (0, 255, 0), 1)  # 第三个参数为线条颜色，第四个参数线条粗度
#             # 100毫秒
#
#             result = []
#             for i in range(len(contours)):
#                 if 1800 > cv2.contourArea(contours[i]) > 150:
#                     # cv2.drawContours(cv2.imread(filepath1) , contours, i, (0, 255, 0), 1)   # i 表示绘制第i条轮廓
#                     x, y, w, h = cv2.boundingRect(contours[i])  # 用一个最小的矩形，把找到的形状包起来
#                     result.append([x, x + w, y, y + h])
#                     img3 = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
#
#             if len(result) == 4:
#                 i = 0
#                 result.sort(key=(lambda x: x[0]))
#                 for [x, x1, y, y1] in result:
#                     mask = 255 - mask
#                     roi = mask[y:y1, x:x1]
#                     # print(name)
#                     if os.path.exists('{0}/{1}'.format(savePath_gen4, name[i])) is False:
#                         os.makedirs('{0}/{1}'.format(savePath_gen4, name[i]))
#                     fileSavePath = '{0}/{1}/{2}{3}.jpg'.format(savePath_gen4, name[i],
#                                                                path + "_" + getname(name) + "_" + name[i], i)
#
#                     print(fileSavePath)
#                     roi = cv2.resize(roi, (30, 30))
#                     cv2.imwrite(fileSavePath, roi)
#                     i += 1
#                 cv2.imshow("image", img)
#                 cv2.waitKey(100)  # 100毫秒
#     else:
#         continue

for path in testPathList:
    # CNN
    if path == "CNN":
        # 打乱随机选200张
        files = os.listdir(testFilePath + path + "/")
        print(len(files))
        random.shuffle(files)
        for fileName in files[:201]:
            if fileName == ".DS_Store":
                continue
            name = fileName.split(".")[0]
            newname = []
            for n in name:
                if n in Big:
                    n = n + n
                newname.append(n)
            name = newname

            filePathIm = testFilePath + path + "/" + fileName
            img = cv2.imread(filePathIm)  # 读入图片
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 将图片转成灰度图
            lower = np.array([0, 0, 160])
            upper = np.array([244, 65, 255])
            mask = cv2.inRange(hsv, lower, upper)
            ret1, im_inv1 = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY_INV)  # 二值化
            kernel1 = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [2, 4, 2]])
            im_blur1 = cv2.filter2D(im_inv1, -1, kernel1)
            # ret2, im_inv2 = cv2.threshold(im_blur1, 200, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(im_blur1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(img, contours, -1, (0, 255, 0), 1)  # 第三个参数为线条颜色，第四个参数线条粗度
            # 100毫秒

            result = []
            for i in range(len(contours)):
                if 4000 > cv2.contourArea(contours[i]) > 200:
                    # cv2.drawContours(cv2.imread(filepath1) , contours, i, (0, 255, 0), 1)   # i 表示绘制第i条轮廓
                    x, y, w, h = cv2.boundingRect(contours[i])  # 用一个最小的矩形，把找到的形状包起来
                    result.append([x, x + w, y, y + h])
                    img3 = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    # cv2.imshow("image", img3)
                    # cv2.waitKey(200)


            # if len(result) <= 4 and len(result) >= 2: # 可以放宽分割条件，增加数据量，但需要手动调整数据集
            if len(result) == 4:
                i = 0
                result.sort(key=(lambda x: x[0]))
                for [x, x1, y, y1] in result:
                    mask = 255 - mask
                    roi = mask[y:y1, x:x1]
                    # print(name)
                    if os.path.exists('{0}/{1}'.format(savePath_CNN, name[i])) is False:
                        os.makedirs('{0}/{1}'.format(savePath_CNN, name[i]))
                    fileSavePath = '{0}/{1}/{2}{3}.jpg'.format(savePath_CNN, name[i],
                                                               path + "_" + getname(name) + "_" + name[i], i)

                    print(fileSavePath)
                    roi = cv2.resize(roi, (30, 30))
                    cv2.imwrite(fileSavePath, roi)
                    i += 1
                cv2.imshow("image", img)
                cv2.waitKey(100)  # 100毫秒

    # wiki
    if path == "wiki":
        files = os.listdir(testFilePath+path+"/")
        random.shuffle(files)
        for fileName in files[:201]:
            if fileName == ".DS_Store":
                continue
            name = fileName.split(".")[0]
            newname = []
            for n in name:
                if n in Big:
                    n = n + n
                newname.append(n)
            name = newname
            filePathIm = testFilePath + path + "/" + fileName
            im1 = cv2.imread(filePathIm)  # 读入图片
            im_gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)  # 将图片转成灰度图

            # 报错 error: (-215:Assertion failed) 原因是文件路径错误
            ret1, im_inv1 = cv2.threshold(im_gray1, 200, 255, cv2.THRESH_BINARY_INV)  # 二值化
            #
            # 应用高斯模糊对图片进行降噪，高斯模糊的本质是用高斯核和图像做卷积
            kernel1 = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [2, 4, 2]])
            im_blur1 = cv2.filter2D(im_inv1, -1, kernel1)
            # 降噪后再做一轮二值化处理
            ret2, im_inv2 = cv2.threshold(im_blur1, 200, 255, cv2.THRESH_BINARY)
            # cv2.imshow("image", im_gray1)
            # cv2.waitKey(100)
            # 把最开始的图片切割成单个字符
            # 第一步 用opencv的findContours来提取轮廓 （cv2.findContours()函数接受的参数为二值图）
            contours, hierarchy = cv2.findContours(im_inv2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 第一个参数是寻找轮廓的图像，第二个参数表示轮廓的检索模式，第三个参数method为轮廓的近似办法
            # cv2.drawContours(im1, contours, -1, (0, 255, 0), 1)  # 第三个参数为线条颜色，第四个参数线条粗度

            num = 0
            result = []
            for i in range(len(contours)):
                if 600 > cv2.contourArea(contours[i]) > 140:
                    num += 1
                    # cv2.drawContours(cv2.imread(filepath1) , contours, i, (0, 255, 0), 1)   # i 表示绘制第i条轮廓
                    x, y, w, h = cv2.boundingRect(contours[i])  # 用一个最小的矩形，把找到的形状包起来
                    result.append([x, x + w, y, y + h])
                    im3 = cv2.rectangle(im1, (x, y), (x + w, y + h), (255, 255, 255), 1)
                    # cv2.imshow("image", im3)
                    # cv2.waitKey(100)  # 100毫秒

            # 放宽分割条件，增加数据量，但需要人工调整数据集
            # if len(result) >= 8 and len(result) <= 9:
            if len(result) == 9:
                i = 0
                result.sort(key=(lambda x: x[0]))
                for [x, x1, y, y1] in result:
                    roi = im1[y:y1, x:x1]
                    if os.path.exists('{0}/{1}'.format(savePath_wiki, name[i])) is False:
                        os.makedirs('{0}/{1}'.format(savePath_wiki, name[i]))
                    fileSavePath = '{0}/{1}/{2}{3}.jpg'.format(savePath_wiki, name[i],
                                                               path + "_" + getname(name) + "_" + name[i], i)
                    print(fileSavePath)
                    roi = cv2.resize(roi, (30, 30))
                    cv2.imwrite(fileSavePath, roi)
                    i += 1
                cv2.imshow("image", im3)
                cv2.waitKey(100)  # 100毫秒

    if path == "bjyh":
        for fileName in os.listdir(testFilePath + path + "/"):
            if fileName == ".DS_Store":
                continue
            name = fileName.split(".")[0]
            newname = []
            for n in name:
                if n in Big:
                    n = n + n
                newname.append(n)
            name = newname
            filePathIm = testFilePath + path + "/" + fileName
            img = cv2.imread(filePathIm)  # 读入图片
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 0, 160])
            upper = np.array([244, 65, 255])
            mask = cv2.inRange(hsv, lower, upper)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            result = []
            for i in range(len(contours)):
                if 900 > cv2.contourArea(contours[i]) > 45:
                    # cv2.drawContours(cv2.imread(filepath1) , contours, i, (0, 255, 0), 1)   # i 表示绘制第i条轮廓
                    x, y, w, h = cv2.boundingRect(contours[i])  # 用一个最小的矩形，把找到的形状包起来
                    result.append([x, x + w, y, y + h])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    # cv2.imshow("image", img)
                    # cv2.waitKey(200)

            if len(result) == 4:
                i = 0
                result.sort(key=(lambda x: x[0]))
                for [x, x1, y, y1] in result:
                    mask = 255 - mask
                    roi = mask[y:y1, x:x1]
                    if os.path.exists('{0}/{1}'.format(savePath_bjyh, name[i])) is False:
                        os.makedirs('{0}/{1}'.format(savePath_bjyh, name[i]))
                    fileSavePath = '{0}/{1}/{2}{3}.jpg'.format(savePath_bjyh, name[i],
                                                               path + "_" + getname(name) + "_" + name[i], i)
                    print(fileSavePath)
                    roi = cv2.resize(roi, (30, 30))
                    cv2.imwrite(fileSavePath, roi)
                    i += 1
                cv2.imshow("image", img)
                cv2.waitKey(100)  # 100毫秒
    if path == "msyh":
         for fileName in os.listdir(testFilePath+path+"/"):
             if fileName == ".DS_Store":
                 continue
             name = fileName.split(".")[0]
             newname = []
             for n in name:
                 if n in Big:
                     n = n+n
                 newname.append(n)
             name  = newname
             filePathIm = testFilePath+path+"/"+fileName
             img = cv2.imread(filePathIm)  # 读入图片
             hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
             lower = np.array([0,0,160])
             upper = np.array([244,65,255])
             mask = cv2.inRange(hsv,lower,upper)
             contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
             result = []
             for i in range(len(contours)):
                 if 900>cv2.contourArea(contours[i]) >45:
                     # cv2.drawContours(cv2.imread(filepath1) , contours, i, (0, 255, 0), 1)   # i 表示绘制第i条轮廓
                     x, y, w, h = cv2.boundingRect(contours[i])  # 用一个最小的矩形，把找到的形状包起来
                     result.append([x,x+w,y,y+h])
                     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                     # cv2.imshow("image", img)
                     # cv2.waitKey(200)

             if len(result)==4:
                 i = 0
                 result.sort(key=(lambda x:x[0]))
                 for [x,x1,y,y1] in result:
                     mask =255 - mask
                     roi = mask[y:y1, x:x1]
                     if os.path.exists('{0}/{1}'.format(savePath_msyh, name[i])) is False:
                         os.makedirs( '{0}/{1}'.format(savePath_msyh, name[i]))
                     fileSavePath = '{0}/{1}/{2}{3}.jpg'.format(savePath_msyh, name[i], path+"_"+getname(name)+"_"+name[i], i)
                     print(fileSavePath)
                     roi = cv2.resize(roi, (30, 30))
                     cv2.imwrite(fileSavePath, roi)
                     i+=1
                 cv2.imshow("image", img)
                 cv2.waitKey(100)  # 100毫秒

    if path == "gfyh":
        for fileName in os.listdir(testFilePath + path + "/"):
            if fileName == ".DS_Store":
                continue
            name = fileName.split(".")[0]
            newname = []
            for n in name:
                if n in Big:
                    n = n + n
                newname.append(n)
            name = newname
            filePathIm = testFilePath + path + "/" + fileName
            img = cv2.imread(filePathIm)  # 读入图片
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 0, 160])
            upper = np.array([244, 65, 255])
            mask = cv2.inRange(hsv, lower, upper)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            result = []
            for i in range(len(contours)):
                if 900 > cv2.contourArea(contours[i]) > 45:
                    # cv2.drawContours(cv2.imread(filepath1) , contours, i, (0, 255, 0), 1)   # i 表示绘制第i条轮廓
                    x, y, w, h = cv2.boundingRect(contours[i])  # 用一个最小的矩形，把找到的形状包起来
                    result.append([x, x + w, y, y + h])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    # cv2.imshow("image", img)
                    # cv2.waitKey(100)  # 100毫秒

            if len(result) == 4:
                i = 0
                result.sort(key=(lambda x: x[0]))
                for [x, x1, y, y1] in result:
                    mask = 255 - mask
                    roi = mask[y:y1, x:x1]
                    if os.path.exists('{0}/{1}'.format(savePath_gfyh, name[i])) is False:
                        os.makedirs('{0}/{1}'.format(savePath_gfyh, name[i]))
                    fileSavePath = '{0}/{1}/{2}{3}.jpg'.format(savePath_gfyh, name[i],
                                                               path + "_" + getname(name) + "_" + name[i], i)
                    print(fileSavePath)
                    roi = cv2.resize(roi, (30, 30))
                    cv2.imwrite(fileSavePath, roi)
                    i += 1
                cv2.imshow("image", img)
                cv2.waitKey(100)  # 100毫秒

    else:
        continue
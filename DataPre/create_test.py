#!/usr/bin/python
# -*- coding: UTF-8 -*-

import requests
from bs4 import BeautifulSoup
import random
import time

savePath_bjyh = r'./data/raw_test/bjyh/'
savePath_msyh = r'./data/raw_test/msyh/'
savePath_gfyh = r'./data/raw_test/gfyh/'

for i in range(210):
    # 北京银行
    # url = 'https://ebank.bankofbeijing.com.cn/servlet/BCCBPB.ImageSignServlet?' + str(random.random())
    # 广发银行
    # url = 'https://ebanks.cgbchina.com.cn/perbank/VerifyImage?update=' + str(random.random())
    # 民生银行
    url = 'https://nper.cmbc.com.cn/pweb/GenTokenImg.do?random='+str(random.random())


    print(url)

    # 构造请求头
    headers = {
        'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding':'gzip, deflate, br',
        'Accept-Language':'zh-CN,zh;q=0.9',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36'
    }
    # 发送请求
    res = requests.get(url, headers=headers,verify=False)
    print(res.status_code)
    # 把获取的二进制写成图片
    # 替换存放路径
    with open(savePath_msyh+str(i)+'.jpg', 'wb') as f:
        f.write(res.content)


# 数据集准备 DataPre
## 数据获取
1. 训练数据：
- 调用python的captcha库生成并自动标注的两万张验证码图片，存放路径为/data/raw_train/captcha/，执行脚本 python create_captcha.py
- 利用Java的Patchca类库生成并标注的另外四类验证码图片，分别存放在/data/raw_train/patchca1{2|3|4}/，Java项目为/DataPre/createPatchca
2. 测试数据：利用Python爬虫爬取的五类真实验证码图像各200张左右，存放在/data/raw_test/下，执行脚本 python create_test.py

## CycleGAN预处理
1. /CycleGAN/datasets/ 下新建文件夹 'noise2denoise'，里面有三个文件夹存放训练和测试样本
- /trainA/：1000张CNN真实验证码
- /trainB/：1000张Java生成的无干扰验证码图像
- /testA/：测试阶段的模型输入，即待处理图片
2. 模型输出即预处理后的验证码图片如“2a3x_fake.png”存放在 /CycleGAN/results/noise2denoise/test_latest/images/下
3. 执行 python  /CycleGAN/changeDIR.py 脚本，将处理后的CNN验证码图片放到/DataPre/data/raw_test/CNN/文件夹下

## 通用预处理+轮廓识别分割
1. 执行脚本 python prePro.py
2. 预处理、分割后的训练和测试字符图像数据存放在 /data/processed_train/ 和 /data/processed_test/ 下
3. 执行脚本 python createMDD.py 构建多源域训练数据集  /data/processed_train/gen5/

# 实验 CRFSL
## 数据集划分
1. 将/DataPre/data/processed_train/下的captcha和gen5两个训练数据集和/DataPre/data/processed_test/下的五个测试数据集放到/CRFSL/filelists/文件夹下
2. 执行脚本 python split_dataset.py 划分训练、测试和验证集，并写入.json文件
- 数字、大小写字母共62类，由于元学习要求训练、测试样本类别不交叉，训练数据集中选46类、测试数据集中选16类。
- 由于每类测试验证码字符分布不均衡，所以先根据分割后的测试数据集选出样本数量前16为的构成测试集类别，从训练数据集中选出剩余46类。
3. 修改 configs.py 下的数据集路径

## 元训练阶段
1. python ./train.py --dataset cnn{wiki | bjyh | msyh | gfyh} --method protonet{maml | protonet_dw} --n_shot 5{1|10} --trainset_aug False{True}
- dataset 参数：选择数据集
- method 参数：选择模型 maml、protonet、protonet_dw（对原型网络距离函数进行改进后模型）
- n_shot 参数：设置小样本学习任务为 5-way n-shot
- trainset_aug 参数：设置是否进行多源域数据增强
2. 保存训练好的模型
- python ./save_features.py --dataset cnn{wiki | bjyh | msyh | gfyh} --method protonet{maml | protonet_dw} --n_shot 5{1|10} --trainset_aug False{True}
3. 模型代码在 ./methods/ 下

## 元测试阶段
python ./test.py --dataset cnn{wiki | bjyh | msyh | gfyh} --method protonet{maml | protonet_dw} --n_shot 5{1|10} --trainset_aug False{True}



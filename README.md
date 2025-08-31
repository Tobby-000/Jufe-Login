# Jufe-Login:一个用于模拟江财Cas登录的Python脚本
江财老Cas登录模拟api,用于验证江财统一认证账密的正确性.内置简易的专用于识别验证码的CNN网络

## 各个脚本或文件的用途
- train.py :训练脚本
- model.py :模型定义脚本
- generate.py :老的用于生成验证码的脚本,已经弃用
- common.py :主要入口以及验证码字符定义
- lookup.py :可视化卷积网络脚本
- predict.py :预测,也就是测试用脚本
- api.py :老Api,已经弃用
- newapi.py:正式使用的api,请训练完模型后以此脚本为入口
- test:验证集
- train.zip:训练集
- mydatasets.py:两个都是用于读取数据以及预处理的,一个老一个新
- jc_model.pth:专用于江财老cas验证码的模型

## 注意
本项目仅用作交流学习,并且已经失效了.

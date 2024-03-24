# 单张测试(数据集中的图片)

import torchvision.transforms as transforms
from PIL import Image
from train import CNN
import imutils
import torch
import cv2
import os

preprocess_transform = transforms.Compose([
    transforms.ToTensor(),
])## 对传入的图片进行处理，转成张量
# 加载模型
model = torch.load('./model/model.pth')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# 识别
def model_test(path):
    image = Image.open(path)#使用PIL加载图像
    image = preprocess_transform(image).unsqueeze_(0)#对图像进行预处理
    outputs = model(image.to(device))
    predict = torch.max(outputs, 1)[1].data.squeeze().item()#对预测结果进行处理
    return predict
#主函数
if __name__ == '__main__':
    files = os.listdir('./test/')#遍历test文件夹
    for file in files:#对列表中的元素进行遍历
        if file.endswith('png'):#判断文件的类型
            path = './test/' + file#图片地址
            out = model_test(path)#调用函数识别
            print(out)

            image = cv2.imread(path)#使用opencv读取图片
            image = imutils.resize(image, width=450)#改变大小
            cv2.imshow('', image)#显示图片
            cv2.waitKey(0)


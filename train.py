
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
import time

# 数据的加载，torchvision.datasets.MNIST是对MNIST数据集的处理，
train_dataset = dsets.MNIST(root='./MNIST_data/',
                            train=True,
                            transform=transforms.ToTensor(),
           download = True)#数据集存在  为False

val_dataset = dsets.MNIST(root='./MNIST_data/',
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True)

#载入数据集 训练的时候选择训练数据集，验证的时候选择验证数据集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)#一次传送的数据集个数100个
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=100)
#关于网络的建立
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)#卷积核的大小   5*5的矩形块，对图片做卷积处理 stride=1 卷积核一次向右或者向下移动一个单位
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)# 池化作用：减小输出大小和降低过拟。降低过拟合是减小输出大小的结果，它同样也减少了后续层中的参数的数
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)#第二卷积层  padding=2在图片的四周补零
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fcl = nn.Linear(32 * 7 * 7, 10)
# 将网络关联起来，构成神经网络
    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fcl(out)
        return out

model = CNN()# 声明一个卷积神经网络
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)# 优化器  lr表示神经网络的学习率
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#
model.to(device)
# 训练和验证过程
def train():

    EPOCH = []#放置迭代次数的列表
    LOSS_train = []#放置训练损失函数的列表
    LOSS_val = []#放置验证损失函数的列表
    ACC_train = []# 放置训练准确率的列表
    ACC_val = []# 放置验证准确率的列表

    for epoch in range(30):

        EPOCH.append(epoch)# 将迭代次数加入到列表中
        start = time.time()#一个迭代的开始时间
        train_loss = 0#初始化三个计数器
        train_correct = 0
        train_count = 0

        for i, (images, labels) in enumerate(train_loader):#得到训练数据集的图片和对应的标签
            images = images.to(device)# 将数据放入指定的设备
            labels = labels.to(device)
            optimizer.zero_grad()# 将梯度置零  loss关于weight的导数变成0
            outputs = model(images)
            loss = criterion(outputs, labels)#使用损失函数，对比预测结果和标签
            loss.backward()#将数据反向传递，改变神经网络的权重以及偏置
            optimizer.step()#用在每个mini-batch之中，进行一步步训练
            pred = outputs.data.max(1, keepdim=True)[1]# 得到识别的结果
            train_correct += pred.eq(labels.data.view_as(pred)).sum()#正确的数量累加
            train_loss += loss.item()#损失函数累加
            train_count += 1# 计数器+1

        LOSS_train.append(train_loss / train_count)#将一个迭代平均损失函数值加入到列表
        ACC_train.append(train_correct / train_count)# 将一个迭代平均准确率加入到列表

        end = time.time()#训练一个迭代终止时间
        print('Epoch: ', epoch + 1)# 输出迭代数
        print('Train Time: ', end - start)#输出训练时间


        # 输出训练准确率和loss

        print('\nTrain-Accuracy: {}/{} ({:.3f}%)\nTrain-Loss: ({:.3f})'.format(train_correct,
                                                                               len(train_loader.dataset),
                                                                               train_correct / train_count,
                                                                               train_loss / train_count))

        # 验证部分
        # 初始化三个计数器
        val_loss = 0
        val_correct = 0
        val_count = 0

        for data, target in val_loader:# 得到验证数据集的图片和对应的标签
            with torch.no_grad():
                data, target = data.to(device), target.to(device),
                outputs = model(data)
                loss = criterion(outputs, target)# 使用损失函数，对比预测结果和标签
                pred = outputs.data.max(1, keepdim=True)[1]#得到识别的结果
                val_correct += pred.eq(target.data.view_as(pred)).sum()#正确的数量累加
                val_loss += loss.item()#损失函数累加
                val_count += 1# 计数器+1

        LOSS_val.append(val_loss / val_count)# 将一个迭代平均损失函数值加入到列表
        ACC_val.append(val_correct / val_count)# 将一个迭代平均准确率加入到列表

        # 输出验证准确率和loss

        print('\nval-Accuracy: {}/{} ({:.3f}%)\nval-Loss: ({:.3f})'.format(val_correct,
                                                                             len(val_loader.dataset),
                                                                             val_correct / val_count,
                                                                             val_loss / val_count))


        # 模型保存
        torch.save(model, './model/model.pth')
        # 绘制图像
        plt.figure()
        plt.plot(EPOCH, LOSS_train)
        plt.title('train loss')
        plt.savefig('./out/train-loss.png')
        plt.close()

        plt.figure()
        plt.plot(EPOCH, LOSS_val)
        plt.title('val loss')
        plt.savefig('./out/val-loss.png')
        plt.close()

        plt.figure()
        plt.plot(EPOCH, ACC_train)
        plt.title('train accuracy')
        plt.savefig('./out/train-acc.png')
        plt.close()

        plt.figure()
        plt.plot(EPOCH, ACC_val)
        plt.title('val accuracy')
        plt.savefig('./out/val-acc.png')
        plt.close()

# 主函数
if __name__ == '__main__':
    train()


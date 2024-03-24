
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from train import CNN
import torch.nn as nn
import torch

test_dataset = dsets.MNIST(root='./MNIST_data/',
                           train=False,
                           transform=transforms.ToTensor(),
                           download = False)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)



model = torch.load('./model/model.pth')#加载模型
model.eval()#启用测试模式
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)#将网络加载到计算设备
criterion = nn.CrossEntropyLoss()

# 初始化三个计数器

test_loss = 0
test_correct = 0
test_count = 0

for data, target in test_loader:#得到测试数据集的图片和对应的标签
    with torch.no_grad():
        data, target = data.to(device), target.to(device),# 将数据放入指定的设备中
        outputs = model(data)
        loss = criterion(outputs, target)#损失函数，对比预测结果和标签
        pred = outputs.data.max(1, keepdim=True)[1]
        test_correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss += loss.item()
        test_count += 1

print('\nTest-Accuracy: {}/{} ({:.3f}%)\nTest-Loss: ({:.3f})'.format(test_correct,
                                                                     len(test_loader.dataset),
                                                                     test_correct / test_count,
                                                                     test_loss / test_count))
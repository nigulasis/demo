import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn


# 1.准备数据：mnist为灰度图，大小28x28，有60，000张训练图像和10，000张测试图像，是图像识别中最为基础的数据集之一。
def prepare_data(batch_size_train=64, batch_size_test=1000):
    """
    数据下载及处理:
    1.ToTensor():将图像从(w,h,c)转为(c,w,h)的tensor格式,同时每个数值除以255，将值域[0,255]归一化为[0,1]。#最大最小归一化:x= x-min(x) / max(x)-min(x)
    2.Normalize()标准化:x=(x-mean)/std   0.1307和0.3081是MNIST数据集的全局平均值和标准偏差,预先计算得到。
    3.DataLoader加速数据读取，shuffle是否打乱数据。
    详细来说：
    1.假设有两个点x1属于[0,1]和x2属于[0,255],激活p=x1*w1+x2*w2,当x2远大于x1时，则激活p受x2的影响将远大于x1，小数值将被忽略，因此载入数据前通常会将各数值转到同一数量级上。
    2.机器学习中，一般希望输入数据是独立同分的，否则网络在学习过程中，上层参数需要不断适应新的输入数据分布，降低学习速度，同时每层的更新都会影响到其它层。将数据转变为标准正态分布更有利于学习
    关于归一化和标准化：都是为方便计算，加速模型的收敛
    """
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True)

    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                           ]))
    test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader


# 2.构建网络
class Model(nn.Module):
    """两层卷积池化+两层全连接"""

    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=3),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2)
                                    )

        self.fc = nn.Sequential(nn.Linear(64 * 5 * 5, 256),
                                nn.ReLU(),
                                nn.Linear(256, 10))

    def forward(self, x):
        # 卷积后图像大小计算公式: N=[(w-k+2p)/s] + 1
        x = self.layer1(x)  # {[(28-3+0)/1]+1}/2 = 13
        x = self.layer2(x)  # {[(13-3+0)/1]+1}/2 = 5 （向下取整）, 卷积+池化后 图像(64,1,28,28)---->(64,64,5,5)
        x = x.reshape(x.size(0), -1)  # (64,64*5*5)
        out = self.fc(x)
        return out


# 3.训练和测试:
def train(epoch, model, train_loader, criterion, optimizer):
    # ==== 一次训练 =====
    model.train()
    train_loss = 0
    train_acc = 0
    for _, (img, label) in enumerate(train_loader):
        if torch.cuda.is_available():
            img, label = img.cuda(), label.cuda()
            model.cuda()

        optimizer.zero_grad()  # 梯度初始化为0
        out = model(img)
        loss = criterion(out, label)  # 计算损失
        loss.backward()  # 反向传播求梯度
        optimizer.step()  # 更新参数

        train_loss += loss.item() * label.size(0)
        _, pred = out.max(1)
        num_correct = pred.eq(label.view_as(pred)).sum()
        train_acc += num_correct.item()

        # =====模型保存=======
        torch.save(model.state_dict(), './model.pth')
        torch.save(optimizer.state_dict(), './optimizer.pth')

    print('Finish  {}  Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, train_loss / len(train_loader),
                                                         train_acc / len(train_loader)))


def test(model, test_loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for img, label in test_loader:
            if torch.cuda.is_available():
                img, label = img.cuda(), label.cuda()
                model.cuda()

            out = model(img)
            _, pred = out.max(1)
            correct += pred.eq(label.view_as(pred)).sum().item()
    print("Test Accuracy: {}%".format(correct / len(test_loader.dataset) * 100))


# ====整合=====
def main():
    '''超参数'''
    iteration_num = 5  # 迭代次数
    batch_size = 64  # 数据的批次大小
    learning_rate = 0.001  # 模型的学习率
    network = Model()  # 实例化网络
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)  # 优化器
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = prepare_data(batch_size)  # 获取数据
    for epoch in range(iteration_num):
        print("\n================ epoch: {} ================".format(epoch))
        train(epoch, network, train_loader, criterion, optimizer)
        test(network, test_loader)


def retrain():
    """提高准确率的常用做法：1.更改训练网络 2.提高训练时长"""
    iteration_num = 5  # 迭代次数
    batch_size = 64  # 数据的批次大小
    learning_rate = 0.0001  # 模型的学习率
    network = Model()  # 实例化网络
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)  # 优化器
    criterion = nn.CrossEntropyLoss()
    # ====加载上次训练模型====
    network_state_dict = torch.load('model.pth')
    optimizer_state_dict = torch.load('optimizer.pth')
    network.load_state_dict(network_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    train_loader, test_loader = prepare_data(batch_size)
    for epoch in range(iteration_num):
        print("\n================new_epoch: {} ================".format(epoch))
        train(epoch, network, train_loader, criterion, optimizer)
        test(network, test_loader)


def show_pic():
    import matplotlib.pyplot as plt

    _, test_loader = prepare_data()
    network = Model()
    network.load_state_dict(torch.load('model.pth'))

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    with torch.no_grad():
        output = network(example_data)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)  # 2行3列子图中的第i+1个位置
        plt.tight_layout()  # 自动调整子图参数
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("预测: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main()
    #retrain()
    #show_pic()

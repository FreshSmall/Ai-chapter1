import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F
import numpy as np
from data_process import get_MNIST_dataloader


# 定义mnist数据识别网络结构，同房价预测网络
class MINIST(paddle.nn.Layer):
    def __init__(self):
        super(MINIST, self).__init__()

        # 定义一层全连接层，输出维度是1
        self.fc = paddle.nn.Linear(in_features=5, out_features=1)

    # 定义网络结构的前向计算过程
    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs


# 定义卷积神经网络结构
class CNN_NET(paddle.nn.Layer):
    def __init__(self):
        super(CNN_NET, self).__init__()
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv1 = Conv2D(in_channels=10, out_channels=20, kernel_size=3, stride=2, padding=1)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 定义一层全连接层，输出维度是1
        # self.fc = Linear(in_features=980, out_features=2)
        # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
        # 卷积层激活函数使用Relu，全连接层不使用激活函数

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        # x = self.fc(x)
        return x


def print_net_summery():
    model = CNN_NET()
    params_info = paddle.summary(model, (100, 10, 7, 7))
    print(params_info)


if __name__ == '__main__':
    print_net_summery()
    # model = CNN_NET()
    # model.train()
    # train_loader, _ = get_MNIST_dataloader()
    # array_1 = np.random.choice((0, 1), size=[64, 1, 28, 28])
    # images = paddle.to_tensor(array_1, dtype='float32')
    # print(model(images))
    # for batch_id, data in enumerate(train_loader()):
    #     # images = data
    #     # images = paddle.to_tensor(images)
    #     # predicts = model(images)
    #     # print(predicts)
    #     # 准备数据
    #     images = data[0]
    #     print(images)
    #     # images = paddle.to_tensor(images)
    #     # labels = paddle.to_tensor(labels, dtype="float32")
    #
    #     # 前向计算的过程
    #     predicts = model(images)  # [batch_size, 1]
    #     print(predicts)
    #     break

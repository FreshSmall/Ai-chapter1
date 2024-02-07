import paddle
from paddle.vision.transforms import Normalize


def get_MNIST_dataloader():
    # 定义图像归一化处理方法，这里的CHW指图像格式需为 [C通道数，H图像高度，W图像宽度]
    transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')
    # 下载数据集并初始化 DataSet
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
    test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

    # 定义并初始化数据读取器
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1, drop_last=True)
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1, drop_last=False)

    return train_loader, test_loader


if __name__ == '__main__':
    print(paddle.vision.datasets.__all__)
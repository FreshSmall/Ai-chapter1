import paddle
from mnist import MINIST
from img import norm_img
import paddle.nn.functional as F

# 确保从paddle.vision.datasets.MNIST中加载的图像数据是np.ndarray类型
paddle.vision.set_image_backend('cv2')

model = MINIST()


def train(model):
    # 启动训练模式
    model.train()
    # 加载训练集 batch_size 设为 16
    train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'),
                                        batch_size=16,
                                        shuffle=True)
    # 定义优化器，使用随机梯度下降SGD优化器，学习率设置为0.001
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    EPOCH_NUM = 10
    loss_list = []
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            images = norm_img(data[0]).astype('float32')
            labels = data[1].astype('float32')

            # 前向计算的过程
            predicts = model(images)

            # 计算损失
            loss = F.square_error_cost(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练了1000批次的数据，打印下当前Loss的情况
            if batch_id % 1000 == 0:
                loss = avg_loss.numpy()[0]
                loss_list.append(loss)
                print("epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    return loss_list

loss_list = train(model)
paddle.save(model.state_dict(), './mnist.pdparams')

if __name__ == '__main__':
    pass
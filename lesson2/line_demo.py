from mnist import MINIST
import paddle
import paddle.nn.functional as F

if __name__ == '__main__':
    model = MINIST()
    model.train()
    test_data = []
    label = []
    opt = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())
    for i in range(1, 30):
        a = [i, i, i, i, i]
        test_data.append(a)
        label.append(i + i * 2 + i + i + i)
    for i in range(len(test_data)):
        data = test_data[i]
        y_data = label[i]
        house_tensor = paddle.to_tensor(data, dtype='float32')
        prices = paddle.to_tensor(y_data, dtype='float32')
        predicts = model(house_tensor)
        # 计算损失，损失函数采用平方误差square_error_cost
        loss = F.square_error_cost(predicts, label=prices)
        avg_loss = paddle.mean(loss)

        print(" loss is: {}".format(avg_loss.numpy()))

        # 反向传播，计算每层参数的梯度值
        avg_loss.backward()
        # 更新参数，根据设置好的学习率迭代一步
        opt.step()
        # 清空梯度变量，进行下一轮计算
        opt.clear_grad()

    paddle.save(model.state_dict(), './demo_model.pdparams')

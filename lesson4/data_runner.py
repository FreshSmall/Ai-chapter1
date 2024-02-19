# -*- coding: utf-8 -*-
# LeNet 识别眼疾图片
import numpy as np
import paddle
from data_reader import data_loader,valid_data_loader
import paddle.nn.functional as F
from paddle.vision.models import resnet50

model = resnet50()

class Runner(object):
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # 记录全局最优指标
        self.best_acc = 0

    # 定义训练过程
    def train_pm(self, train_datadir, val_datadir, **kwargs):
        print('start training ... ')
        self.model.train()

        num_epochs = kwargs.get('num_epochs', 0)
        csv_file = kwargs.get('csv_file', 0)
        save_path = kwargs.get("save_path", "/home/aistudio/output/")
        print(save_path)

        # 定义数据读取器，训练数据读取器
        train_loader = data_loader(train_datadir, batch_size=10, mode='train')

        for epoch in range(num_epochs):
            for batch_id, data in enumerate(train_loader()):
                x_data, y_data = data
                img = paddle.to_tensor(x_data)
                label = paddle.to_tensor(y_data)
                # 运行模型前向计算，得到预测值
                logits = model(img)
                avg_loss = self.loss_fn(logits, label)

                if batch_id % 20 == 0:
                    print("epoch: {}, batch_id: {}, loss is: {:.4f}".format(epoch, batch_id, float(avg_loss.numpy())))
                # 反向传播，更新权重，清除梯度
                avg_loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()

            acc = self.evaluate_pm(val_datadir, csv_file)
            self.model.train()
            if acc > self.best_acc:
                self.save_model(save_path)
                self.best_acc = acc

    # 模型评估阶段，使用'paddle.no_grad()'控制不计算和存储梯度
    @paddle.no_grad()
    def evaluate_pm(self, val_datadir, csv_file):
        self.model.eval()
        accuracies = []
        losses = []
        # 验证数据读取器
        valid_loader = valid_data_loader(val_datadir, csv_file)

        for batch_id, data in enumerate(valid_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            # 运行模型前向计算，得到预测值
            logits = self.model(img)
            # 多分类，使用softmax计算预测概率
            pred = F.softmax(logits)
            loss = self.loss_fn(pred, label)
            acc = paddle.metric.accuracy(pred, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))
        return np.mean(accuracies)

        # 模型评估阶段，使用'paddle.no_grad()'控制不计算和存储梯度

    @paddle.no_grad()
    def predict_pm(self, x, **kwargs):
        # 将模型设置为评估模式
        self.model.eval()
        # 运行模型前向计算，得到预测值
        logits = self.model(x)
        return logits

    def save_model(self, save_path):
        paddle.save(self.model.state_dict(), save_path + 'palm.pdparams')
        paddle.save(self.optimizer.state_dict(), save_path + 'palm.pdopt')

    def load_model(self, model_path):
        model_state_dict = paddle.load(model_path)
        self.model.set_state_dict(model_state_dict)

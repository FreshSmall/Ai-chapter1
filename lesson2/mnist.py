import paddle

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
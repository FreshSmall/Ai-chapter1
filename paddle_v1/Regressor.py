# 加载飞桨、NumPy和相关类库
import paddle
from paddle.nn import Linear


class Regressor(paddle.nn.Layer):
    # self 代表类的实例本身
    def __init__(self):
        # 初始化父类中的一些参数
        super(Regressor, self).__init__()
        # 定义一层全连接层，输入维度是13，输出维度是1
        self.fc = Linear(in_features=13, out_features=1)

    # 网络的前向计算
    def forward(self, inputs):
        x = self.fc(inputs)
        return x

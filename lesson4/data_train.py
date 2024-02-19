import os

import paddle
import paddle.nn.functional as F
from paddle.vision.models import resnet50

from data_runner import Runner

loss_fn = F.cross_entropy
model = resnet50()

# 开启0号GPU训练
use_gpu = True
paddle.device.set_device('cpu') if use_gpu else paddle.device.set_device('cpu')

# 定义优化器
# opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters(), weight_decay=0.001)
opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())

runner = Runner(model, opt, loss_fn)

if __name__ == '__main__':
    # 数据集路径
    DATADIR = '/Users/yinchao/python-workspace/plam/PALM-Training400/PALM-Training400'
    DATADIR2 = '/Users/yinchao/python-workspace/plam/PALM-Validation400'
    CSVFILE = '/Users/yinchao/python-workspace/plam/labels.csv'
    # 设置迭代轮数
    EPOCH_NUM = 5
    # 模型保存路径
    PATH = '/Users/yinchao/python-workspace/Ai-chapter1/lesson4/output/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    # 启动训练过程
    runner.train_pm(DATADIR, DATADIR2, num_epochs=EPOCH_NUM, csv_file=CSVFILE, save_path=PATH)

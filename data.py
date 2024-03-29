import numpy as np


class DataSource(object):

    def load_data(self):
        # 从文件导入数据
        datafile = './housing.data'
        data = np.fromfile(datafile, sep=' ')

        # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
        feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                         'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        feature_num = len(feature_names)

        # 将原始数据进行Reshape，变成[N, 14]这样的形状
        data = data.reshape([data.shape[0] // feature_num, feature_num])

        # 将原数据集拆分成训练集和测试集
        # 这里使用80%的数据做训练，20%的数据做测试
        # 测试集和训练集必须是没有交集的
        ratio = 0.8
        offset = int(data.shape[0] * ratio)
        training_data = data[:offset]

        # 计算训练集的最大值，最小值
        maximums, minimums = training_data.max(axis=0), \
                             training_data.min(axis=0)

        # 对数据进行归一化处理
        for i in range(feature_num):
            data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

        # 训练集和测试集的划分比例
        training_data = data[:offset]
        test_data = data[offset:]
        return training_data, test_data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#设置图片大小
plt.figure(figsize=(8, 4))

# x是1维数组，数组大小是从-10. 到10.的实数，每隔0.1取一个点
x = np.arange(-10, 10, 0.1)
# 计算 Sigmoid函数
y = (np.exp(x)-np.exp(-x)) / (np.exp(x) + np.exp(- x))

#########################################################
# 以下部分为画图程序

# 设置两个子图窗口，将Sigmoid的函数图像画在左边
# 画出函数曲线
plt.plot(x, y, color='r')
# 添加文字说明
plt.text(-5., 0.9, r'$y=\tanh(x)$', fontsize=13)

plt.show()

if __name__ == '__main__':
    pass
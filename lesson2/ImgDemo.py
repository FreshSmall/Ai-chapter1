import paddle
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# 图像归一化函数，将数据范围为[0, 255]的图像归一化到[0, 1]
def norm_img(img):
    # 验证传入数据格式是否正确，img的shape为[batch_size, 28, 28]
    assert len(img.shape) == 3
    batch_size, img_h, img_w = img.shape[0], img.shape[1], img.shape[2]
    # 归一化图像数据
    img = img / 255
    # 将图像形式reshape为[batch_size, 784]
    img = paddle.reshape(img, [batch_size, img_h * img_w])

    print(batch_size)
    print(img_h)
    print(img_w)

    return img


# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert('L')
    # print(np.array(im))
    im = im.resize((28, 28), Image.LANCZOS)
    im = np.array(im).reshape(1, -1).astype(np.float32)
    # 图像归一化，保持和数据集的数据范围一致
    im = 1 - im / 255
    return im


if __name__ == '__main__':
    # im = Image.open('/Users/yinchao/example_6.jpg')
    im_path = '/Users/yinchao/example_6.jpg'
    im = load_image(im_path)
    plt.imshow(im)
    plt.show()
    # 将原始图像转为灰度图
    im = im.convert('L')
    print('原始图像shape: ', np.array(im).shape)
    # 使用Image.ANTIALIAS方式采样原始图片
    im = im.resize((28, 28), Image.LANCZOS)
    plt.imshow(im)
    plt.show()
    print("采样后图片shape: ", np.array(im).shape)

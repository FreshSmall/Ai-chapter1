import numpy as np
import paddle


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


if __name__ == '__main__':
    a = np.asarray([3, 1, -3])
    print(softmax(a))
    device = paddle.device.get_device()
    print(device)

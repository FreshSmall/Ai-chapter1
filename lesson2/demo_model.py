import paddle
from mnist import MINIST

if __name__ == '__main__':
    # 将模型参数保存到指定路径中
    model_dict = paddle.load('demo_model.pdparams')
    model = MINIST()
    model.load_dict(model_dict)
    # 将模型状态修改为.eval
    model.eval()
    data = [20, 20, 20, 20, 20]
    one_data = data
    # 将数据格式转换为张量
    one_data = paddle.to_tensor(one_data, dtype="float32")
    predict = model(one_data)
    print(predict)
    # print("Inference result is {}, the corresponding label is {}".format(predict.numpy(), label))

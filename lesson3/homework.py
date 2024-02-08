import paddlehub as hub
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def test6():
    model = hub.Module(name='UGATIT_100w')
    # 结果保存在'output/'目录，可以观察可视化结果
    result = model.style_transfer(images=[cv2.imread('test6.jpg')], visualization=True)

    img_ori = Image.open('test6.jpg')
    img = cv2.cvtColor(result[0], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    fig = plt.figure(figsize=(8, 8))
    # 显示原图
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img_ori)
    # 显示生成漫画图
    ax = fig.add_subplot(1, 2, 2)
    # ax.imshow(img)
    plt.show()


def test1():
    print("开始启动")
    classifier = hub.Module(name="resnet50_vd_dishes")
    print("步骤1")
    result = classifier.classification(images=[cv2.imread('imgs/test1.jpg')])
    print('result:{}'.format(result))


def testLanguage():
    lac = hub.Module(name="lac")
    test_text = ["今天是个好天气。"]
    results = lac.cut(text=test_text, use_gpu=False, batch_size=1, return_tag=True)
    print(results)


if __name__ == '__main__':
    testLanguage()

import paddlehub as hub
import cv2
from PIL import Image
import matplotlib.pyplot as plt

model = hub.Module(name='UGATIT_100w')
# 结果保存在'output/'目录，可以观察可视化结果
result = model.style_transfer(images=[cv2.imread('./imgs/test6.jpg')], visualization=True)

img_ori = Image.open('./imgs/test6.jpg')
img = cv2.cvtColor(result[0], cv2.COLOR_BGR2RGB)
img = Image.fromarray(img)

fig = plt.figure(figsize=(8, 8))
# 显示原图
ax = fig.add_subplot(1, 2, 1)
ax.imshow(img_ori)
# 显示生成漫画图
ax = fig.add_subplot(1, 2, 2)
ax.imshow(img)
plt.show()

if __name__ == '__main__':
    pass

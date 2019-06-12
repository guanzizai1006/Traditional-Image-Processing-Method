import matplotlib.pyplot as plt
import cv2 as cv
from skimage import data,color,morphology,feature


"""
# 生成二值测试图像
img = color.rgb2gray(data.coins())
# 检测canny边缘，得到二值图片
egds = feature.canny(img, sigma=3, low_threshold=10, high_threshold=50)
"""



img = color.rgb2gray(data.horse())
img = (img<0.5)*1

chull = morphology.convex_hull_image(img)

# 绘制轮廓
fig, axes = plt.subplots(2, 1, figsize=(6, 6))
ax0, ax1 = axes.ravel()
ax0.imshow(img, plt.cm.gray)
ax0.set_title('original image')

ax1.imshow(chull, plt.cm.gray)
ax1.set_title('convex_hull image')
# cv.imwrite('./one.jpg', chull)
# cv.imwrite("./horse.jpg", img)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import data,filters,segmentation,measure,morphology,color
# Notes:author写错了，应该是filters而不是filter
# https://www.cnblogs.com/denny402/p/5166258.html

#加载并裁剪硬币图片
image = data.coins()[50:-50, 50:-50]

# thresh = skimage.filter.thresholding.threshold_otsu(image)
thresh =filters.threshold_otsu(image) #阈值分割
bw =morphology.closing(image > thresh, morphology.square(3)) #闭运算

cleared = bw.copy()  #复制
segmentation.clear_border(cleared)  #清除与边界相连的目标物

label_image =measure.label(cleared)  #连通区域标记
borders = np.logical_xor(bw, cleared) #异或
label_image[borders] = -1


image_label_overlay =color.label2rgb(label_image, image=image) #不同标记用不同颜色显示
"""bruce说，这是在skimage.color()模块下的结果，在color模块的颜色空间转换函数中，还有一个比较有用的函数是
skimage.color.label2rgb(arr), 可以根据标签值对图片进行着色。以后的图片分类后着色就可以用这个函数。
例：将lena图片分成三类，然后用默认颜色对三类进行着色	https://www.jianshu.com/p/f2e88197e81d
"""


fig,(ax0,ax1)= plt.subplots(1,2, figsize=(8, 6))
ax0.imshow(cleared,plt.cm.gray)
ax1.imshow(image_label_overlay)

for region in measure.regionprops(label_image): #循环得到每一个连通区域属性集
    
    #忽略小区域
    if region.area < 100:
        continue

    #绘制外包矩形
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
    ax1.add_patch(rect)
    
fig.tight_layout()
plt.show()


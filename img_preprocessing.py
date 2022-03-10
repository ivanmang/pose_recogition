import os
import cv2
import imutils
import numpy as np

path='/data/MultiModHandGestRecog/MultiModHandGestRecog/near-infrared/user_01/train_pose/five/frame_40695_r.png'
img=cv2.imread(path)
dir='./data'

dir ='./data'
bg = None

def crop_resize(path, dir):
    h, w=np.where(img[:, :, 0]>100)
    bottom, top=min(h[0]), max(h[0])
    left, right=min(w[0]), max(w[1])
    roi=img[bottom:top, left:right]
    res=cv2.resize(roi, (32, 32))
    if 'train' in path:
        cv2.imwrite(os.path.join(dir, 'train', path.split('/')[-1]))
    if 'test' in path:
        cv2.imwrite(os.path.join(dir, 'test', path.split('/')[-1]))



def contour_proc(img_path):
    offset=20
    img=cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(gray, 128, 255, 0)
    h, w= np.where(im_bw == 255)
    bottom, top, left, right=h.min(), h.max(), w.min(), w.max()
    roi=img[bottom:top+offset, left-offset:right+offset]
  #  dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
  #binaryimg = cv2.Canny(Laplacian, 50, 200) #二值化，canny检测
   # h = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #寻找轮廓
   # contour = h[0]
   # contour = sorted(contour, key = cv2.contourArea, reverse=True)#已轮廓区域面积进行排序
    #contourmax = contour[0][:, 0, :]#保留区域面积最大的轮廓点坐标
   # bg = np.ones(dst.shape, np.uint8) *255#创建白色幕布
   # ret = cv2.drawContours(bg,contour[0],-1,(0,0,0),3) #绘制黑色轮廓
    return roi




import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split


selected=['five', 'two', 'fist_moved']

path='/data/MultiModHandGestRecog/MultiModHandGestRecog/near-infrared/'
dir ='./data'
print(os.listdir(path))

def crop_resize(img):
    h, w=np.where(img[:, :, 0]>80)
    if h.any()>0 and w.any()>0:
        bottom, top=min(h), max(h)
        left, right=min(w), max(w)
        roi=img[bottom:top, left:right]
        if roi is not None and roi.shape[0]>0 and roi.shape[1]>0:
            print(roi.shape)
            res = cv2.resize(roi, (32, 32))
            if res is not None:
                return res


def contour_proc(img_path):
    offset=20
#    img=cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(gray, 128, 255, 0)
    h, w= np.where(im_bw == 255)
    if len(h)>0 and len(w)>0:
        bottom, top, left, right=h.min(), h.max(), w.min(), w.max()
        roi=img[bottom:top+offset, left-offset:right+offset]
        if roi is not None and roi.shape[0] > 0 and roi.shape[1] > 0:
            print(roi.shape)
            res = cv2.resize(roi, (32, 32))
            if res is not None:
                return res



train_set = []
for user in os.listdir(path):
    for clss in selected:
        for img in os.listdir(os.path.join(path, user, 'train_pose', clss)):
            train_set.append(os.path.join(path, user, 'train_pose', clss, img))

train_set, dev_set= train_test_split(train_set, test_size=0.2, random_state=42)


for user in os.listdir(path):
    for t in ['test_pose']:
        for clss in selected:
            for img in os.listdir(os.path.join(path, user, t, clss)):
                path_img=os.path.join(path, user, t, clss, img)
                img=cv2.imread(path_img)
                res_img=contour_proc(img)
                _=path_img.split('/')[-3:]
                try:
                    cv2.imwrite(os.path.join(dir, _[0], _[1], _[2]), res_img)
                except:
                    pass
                else:
                    cv2.imwrite(os.path.join(dir, _[0], _[1], _[2]), res_img)




for i in train_set:
    img = cv2.imread(i)
    res_img = contour_proc(img)
    _ = i.split('/')[-3:]
    if res_img is not None:
        cv2.imwrite(os.path.join(dir, _[0], _[1], _[2]), res_img)
for i in dev_set:
    img = cv2.imread(i)
    res_img = contour_proc(img)
    _ = i.split('/')[-3:]
    if res_img is not None:
        cv2.imwrite(os.path.join(dir, 'dev_pose', _[1], _[2]), res_img)








print()
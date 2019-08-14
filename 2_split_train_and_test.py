import os
import cv2
import shutil
import random

"""This file is for spliting the training datas and test datas randomly. And then we split big picture into small picture"""
def split_train_and_test(imgs_dir, labels_dir, test_dir='test', test_rate=0.15):  # split train and test datas.
    test_img = os.path.join(test_dir, 'imgs')
    os.makedirs(test_img, exist_ok=True)
    test_label = os.path.join(test_dir, 'labels')
    os.makedirs(test_label, exist_ok=True)
    img_list = os.listdir(imgs_dir)
    random.shuffle(img_list)
    for i in range(int(len(img_list) * test_rate)):
        shutil.move(os.path.join(imgs_dir, img_list[i]), os.path.join(test_img, img_list[i]))
        shutil.move(os.path.join(labels_dir, img_list[i]), os.path.join(test_label, img_list[i]))

def split_datas(img_dir, label_dir, save_img_dir, save_label_dir):  # split big images into small images.
    os.makedirs(save_img_dir,exist_ok=True)
    os.makedirs(save_label_dir,exist_ok=True)
    imgs = os.listdir(img_dir)
    num_img = 0
    for i in range(len(imgs)):
        img_arr = cv2.imread(os.path.join(img_dir, imgs[i]))
        h, w = img_arr.shape[:2]
        label_arr = cv2.imread(os.path.join(label_dir, imgs[i]))
        cv2.imwrite(os.path.join(save_img_dir,str(num_img)+'.jpg'), img_arr[int(h/4):int(3*h/4), int(w/4):int(3*w/4) ])
        cv2.imwrite(os.path.join(save_label_dir,str(num_img)+'.jpg'), label_arr[int(h/4):int(3*h/4), int(w/4):int(3*w/4)])
        num_img += 1
        cv2.imwrite(os.path.join(save_img_dir,str(num_img)+'.jpg'), img_arr[0:int(h/2), 0:int(w/2), ])
        cv2.imwrite(os.path.join(save_label_dir,str(num_img)+'.jpg'), label_arr[0:int(h/2), 0:int(w/2)])
        num_img += 1
        cv2.imwrite(os.path.join(save_img_dir,str(num_img)+'.jpg'), img_arr[0:int(h/2), int(w/2):])
        cv2.imwrite(os.path.join(save_label_dir,str(num_img)+'.jpg'), label_arr[0:int(h/2), int(w/2):])
        num_img += 1
        cv2.imwrite(os.path.join(save_img_dir,str(num_img)+'.jpg'), img_arr[int(h/2):, 0:int(w/2)])
        cv2.imwrite(os.path.join(save_label_dir,str(num_img)+'.jpg'), label_arr[int(h/2):, 0:int(w/2)])
        num_img += 1
        cv2.imwrite(os.path.join(save_img_dir,str(num_img)+'.jpg'), img_arr[int(h/2):, int(w/2):])
        cv2.imwrite(os.path.join(save_label_dir,str(num_img)+'.jpg'), label_arr[int(h/2):, int(w/2):])
        num_img += 1
    print('Split datas Finished')

if __name__ == '__main__':
    img_dir = r'source/imgs'
    label_dir = r'source/labels'
    save_img_dir = r'train/imgs'
    save_label_dir = r'train/labels'
    split_datas(img_dir, label_dir, save_img_dir, save_label_dir)
    split_train_and_test(save_img_dir, save_label_dir, test_dir='test', test_rate=0.15)
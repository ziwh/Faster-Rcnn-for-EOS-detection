import cv2
import os
import shutil
import numpy as np
import random
import copy
from keras.preprocessing.image import ImageDataGenerator
from tools import *

def split_and_flip_aug(img_path, label_path, aug_img_path, aug_label_path): # augmentation by split into samll images and flip
    os.makedirs(aug_img_path, exist_ok=True)
    os.makedirs(aug_label_path, exist_ok=True)
    img_list = os.listdir(img_path)
    for i in range(len(img_list)):
        img = cv2.imread(os.path.join(img_path, img_list[i]))
        label = cv2.imread(os.path.join(label_path, img_list[i]))
        h, w = img.shape[:2]
        crop_xmin = random.randint(0, int(h*0.2))
        crop_ymin = random.randint(0, int(w*0.2))
        crop_xmax = random.randint(int(h*0.8), h)
        crop_ymax = random.randint(int(w*0.8), w)
        img1 = cv2.resize(img[:360, :480], (w, h))
        label1 = cv2.resize(label[:360,:480], (w, h))
        img2 = cv2.resize(img[:360,480:], (w, h))
        label2 = cv2.resize(label[:360,480:], (w, h))
        img3 = cv2.resize(img[360:,:480], (w, h))
        label3 = cv2.resize(label[360:,:480], (w, h))
        img4 = cv2.resize(img[360:,480:], (w, h))
        label4 = cv2.resize(label[360:,480:], (w, h))
        flip_way = random.randint(-1, 1)
        aug_img1 = cv2.resize(cv2.flip(img1[crop_xmin:crop_xmax, crop_ymin:crop_ymax], flip_way), (w, h))
        cv2.imwrite(os.path.join(aug_img_path, str(i * 8) + '.jpg'), aug_img1)
        aug_label1 = cv2.resize(cv2.flip(label1[crop_xmin:crop_xmax, crop_ymin:crop_ymax], flip_way), (w, h))
        cv2.imwrite(os.path.join(aug_label_path, str(i * 8) + '.jpg'), aug_label1)
        aug_img2 = cv2.resize(cv2.flip(img2[crop_xmin:crop_xmax, crop_ymin:crop_ymax], flip_way), (w, h))
        cv2.imwrite(os.path.join(aug_img_path, str(i * 8 + 1) + '.jpg'),aug_img2)
        aug_label2 = cv2.resize(cv2.flip(label2[crop_xmin:crop_xmax, crop_ymin:crop_ymax], flip_way), (w, h))
        cv2.imwrite(os.path.join(aug_label_path, str(i * 8 + 1) + '.jpg'), aug_label2)
        aug_img3 = cv2.resize(cv2.flip(img3[crop_xmin:crop_xmax, crop_ymin:crop_ymax], flip_way), (w, h))
        cv2.imwrite(os.path.join(aug_img_path, str(i * 8 + 2) + '.jpg'), aug_img3)
        aug_label3 = cv2.resize(cv2.flip(label3[crop_xmin:crop_xmax, crop_ymin:crop_ymax], flip_way), (w, h))
        cv2.imwrite(os.path.join(aug_label_path, str(i * 8 + 2) + '.jpg'), aug_label3)
        aug_img4 = cv2.resize(cv2.flip(img4[crop_xmin:crop_xmax, crop_ymin:crop_ymax], flip_way), (w, h))
        cv2.imwrite(os.path.join(aug_img_path, str(i * 8 + 3) + '.jpg'), aug_img4)
        aug_label4 = cv2.resize(cv2.flip(label4[crop_xmin:crop_xmax, crop_ymin:crop_ymax], flip_way), (w, h))
        cv2.imwrite(os.path.join(aug_label_path, str(i * 8 + 3) + '.jpg'), aug_label4)
        cv2.imwrite(os.path.join(aug_img_path, str(i * 8 + 4) + '.jpg'), img1)
        cv2.imwrite(os.path.join(aug_img_path, str(i * 8 + 5) + '.jpg'), img2)
        cv2.imwrite(os.path.join(aug_img_path, str(i * 8 + 6) + '.jpg'), img3)
        cv2.imwrite(os.path.join(aug_img_path, str(i * 8 + 7) + '.jpg'), img4)
        cv2.imwrite(os.path.join(aug_label_path, str(i * 8 + 4) + '.jpg'), label1)
        cv2.imwrite(os.path.join(aug_label_path, str(i * 8 + 5) + '.jpg'), label2)
        cv2.imwrite(os.path.join(aug_label_path, str(i * 8 + 6) + '.jpg'), label3)
        cv2.imwrite(os.path.join(aug_label_path, str(i * 8 + 7) + '.jpg'), label4)
    print("***********Split end**************")

def objects_and_imgs_aug(imgs_dir, labels_dir, masks_dir='results/masks'):  # augmentation those images with many eos cells and copy eos cells into some picture without eos
    os.makedirs(masks_dir,exist_ok=True)
    img_areas, label_areas, without_cell_imgs = [], [], []
    par = 90
    dia = 45
    img_nums = 20000
    img_list = os.listdir(imgs_dir)
    print("*************make masks*******************")
    for item in img_list:
        img_arr = cv2.imread(os.path.join(imgs_dir, item))
        label_arr = cv2.imread(os.path.join(labels_dir, item))
        points = label_creator(label_arr)
        mask_out = copy.deepcopy(img_arr)
        if len(points) < 2:
            without_cell_imgs.append(item)
        else:
            for point in points:
                if point[0] - dia > 0 and point[1] - dia > 0 and point[0] + dia < 960 and point[1] + dia < 720:
                    cv2.rectangle(mask_out, (point[0] - dia, point[1] - dia) , (point[0] + dia, point[1] + dia), (0, 255, 0), 2)
                    if random.random() < 0.1:
                        img_areas.append(img_arr[(point[1] - dia):(point[1] + dia),(point[0] - dia) : (point[0] + dia)])
                        label_areas.append(label_arr[(point[1] - dia):(point[1] + dia),(point[0] - dia) : (point[0] + dia)])
            cv2.imwrite(os.path.join(masks_dir ,item), mask_out)
        if len(points) > 3 and random.random() < 0.9:
            flip_status = random.randint(-1, 1)
            aug_img1 = cv2.flip(img_arr, flip_status)
            aug_label1 = cv2.flip(label_arr, flip_status)
            cv2.imwrite(os.path.join(imgs_dir, str(img_nums) + '.jpg'), aug_img1)
            cv2.imwrite(os.path.join(labels_dir, str(img_nums) + '.jpg'), aug_label1)
            img_nums += 1
    print("****************Copy cell targets****************")
    for item in without_cell_imgs:
        img_arr = cv2.imread(os.path.join(imgs_dir, item))
        label_arr = cv2.imread(os.path.join(labels_dir, item))
        y = random.randint(30, 120)
        x = random.randint(30, 120)
        for a in range(4):
            for b in range(3):
                x_b = random.randint(-30, 30)
                y_b = random.randint(-30, 30)
                c = random.randint(0,len(img_areas) - 1)
                img_arr[(y + y_b + b * 180):(y + y_b + par + b * 180), (x + x_b + a * 180):(x + x_b + par + a * 180), :] = img_areas[c]
                label_arr[(y + y_b + b * 180):(y + y_b + par + b * 180), (x + x_b + a * 180):(x + x_b + par + a * 180), :] = label_areas[c]
        cv2.imwrite(os.path.join(imgs_dir, str(img_nums + 10000) + '.jpg'), img_arr)
        cv2.imwrite(os.path.join(labels_dir, str(img_nums + 10000) + '.jpg'), label_arr)
        img_nums += 1

def keras_augmentation(imgs_dir, labels_dir):   # random augmentation by keras.preprocessing.image ImageDataGenerator
    batch_size = 50
    image_shape = (720, 960)
    count = 0
    seed = 1
    img_nums = 50000
    imgs_save_dir = "temp/aug_imgs"
    labels_save_dir = "temp/aug_labels"
    imgs_path = "temp/imgs/datas"
    labels_path = "temp/labels/datas"
    os.makedirs(imgs_save_dir, exist_ok=True)
    os.makedirs(labels_save_dir, exist_ok=True)
    shutil.copytree(imgs_dir, imgs_path)
    shutil.copytree(labels_dir, labels_path)
    data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rescale=1./255,
                     width_shift_range=0.15,
                     rotation_range=90,
                     height_shift_range=0.15,
                     zoom_range=0.15,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='constant',
                     cval=0)
    image_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow_from_directory("temp/imgs",
                                                      batch_size = batch_size,
                                                      target_size = image_shape,
                                                      class_mode=None,
                                                      seed=seed,
                                                      save_to_dir = imgs_save_dir)
    
    label_generator = image_datagen.flow_from_directory("temp/labels",
                                                      batch_size = batch_size,
                                                      target_size = image_shape,
                                                      class_mode=None,
                                                      seed=seed,
                                                      save_to_dir = labels_save_dir)
    
    train_generator = zip(image_generator, label_generator)
    for x_batch_image, x_batch_mask in train_generator:
        print(F"-----------{count}------------")
        count += 1
        if count == 50:
            break
    img_list = os.listdir(imgs_save_dir)
    for item in img_list:
    	shutil.copy(os.path.join(imgs_save_dir, item), os.path.join(imgs_dir, str(img_nums) + '.jpg'))
    	shutil.copy(os.path.join(labels_save_dir, item), os.path.join(labels_dir, str(img_nums) + '.jpg'))
    	img_nums += 1
    # os.removedirs("temp")
    print("************keras augmentataion finished*************")
    

if __name__ == '__main__':
    img_path = 'train/imgs'
    label_path = 'train/labels'
    aug_img_path = 'aug/imgs'
    aug_label_path = 'aug/labels'
    test_imgs = 'test/imgs'
    test_labels = 'test/labels'
    split_and_flip_aug(img_path, label_path, aug_img_path, aug_label_path)
    split_and_flip_aug(test_imgs, test_labels, 'test/aug_imgs', 'test/aug_labels')
    objects_and_imgs_aug(aug_img_path, aug_label_path, masks_dir='results/masks')
    keras_augmentation(aug_img_path, aug_label_path)
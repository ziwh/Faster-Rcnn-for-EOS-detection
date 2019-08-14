import cv2
import sys
import numpy as np
import os
import shutil
from tqdm import tqdm
from tools import *

"""Because many images in the source label images are labeled by green points and black box, so
we decide to change the black boxes to green points to make the later work more sucessful.
This file is used to do this"""
def useful_marks(hsv):	# some area was labeled by green points and black boxes was not useful, so find the uesful points and return 
    points = point_detect(hsv, yellow=False)
    boxes = box_detect(hsv)
    useful_boxes = [box for box in boxes if not any([inside(point, box) for point in points])]
    useful_points = [point for point in points if not any([inside(point, box) for box in boxes])]
    box_points = [middle(box) for box in useful_boxes]
    useful_points.extend(box_points)
    return useful_points

def mark_point(img, points, radius=5):
    green = (0, 255, 0)
    for point in points:
        cv2.circle(img, point, radius, green, -1)
    return img

def point_label_creator(dir_img, dir_label, dst):
    os.makedirs(dst, exist_ok=True)
    img_list = os.listdir(dir_img)
    for item in tqdm(img_list):
    	img_arr = cv2.imread(os.path.join(dir_img, item))
    	label_arr = cv2.imread(os.path.join(dir_label, item))
    	hsv = cv2.cvtColor(label_arr, cv2.COLOR_BGR2HSV)
    	points = useful_marks(hsv)
    	draw = mark_point(img_arr, points)
    	cv2.imwrite(os.path.join(dst ,item), draw)

if __name__ == "__main__":
    dir_img = 'source/imgs'
    dir_label = 'source/labels'
    dst = 'source/true_labels'
    point_label_creator(dir_img, dir_label, dst)
    shutil.move(dir_label, "source/source_labels")
    shutil.move(dst, dir_label)
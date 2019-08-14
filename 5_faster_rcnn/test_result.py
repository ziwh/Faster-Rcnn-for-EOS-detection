import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import numpy as np
import pandas as pd
import cv2
import copy
os.environ['CUDA_VISIIBLE_DEVICES']= '4'


def middle(cnt): # return a object center point
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return (0, 0)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy)

def point_detect(hsv, yellow=False):  # yellow or blue point detection
    """"""
    if yellow:
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
    else:
        lower = np.array([50,100, 50])
        upper = np.array([70,255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    points = [middle(cnt) for cnt in cnts]
    return points

def inside(point, cnt, threshold=-3):
    return cv2.pointPolygonTest(cnt, point, True) >= threshold

def label_creator(label):
    label_hsv = cv2.cvtColor(label, cv2.COLOR_BGR2HSV)
    points = point_detect(label_hsv)
    # labels = watershed_labels(img)
    return points

def test_results(imgs_path, labels_path, results_path, model_path):
   os.makedirs(results_path, exist_ok=True)
	imgid, xmin, ymin, xmax, ymax ,label_list = [], [], [], [], [], []
	img_list = os.listdir(imgs_path)
	all_EOS_cells = 0
	all_green_points = 0
	TP, FP, FN = 0, 0, 0
	faster_rcnn = FasterRCNNVGG16()
	trainer = FasterRCNNTrainer(faster_rcnn).cuda()
	trainer.load(model_path)

	for item in img_list:
	    img = read_image(os.path.join(imgs_path, item))
	    label_arr = cv2.imread(os.path.join(labels_path, item))
	    result_arr = copy.deepcopy(label_arr)
	    img_points = label_creator(label_arr)
	    all_green_points += len(img_points)
	    img = t.from_numpy(img)[None]
	    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
	    for i in range(_bboxes[0].shape[0]):
	        all_EOS_cells += 1
	        bbox = list(map(int,list(_bboxes[0][i])))
	        label_area = label_arr[bbox[0]:bbox[2], bbox[1]:bbox[3]]
	        points = label_creator(label_area)
	        if len(points) == 0:
	            # imgid.append(item.split('.')[0])
	            # label_list.append(0)
	            # xmin.append(bbox[1])
	            # ymin.append(bbox[0])
	            # xmax.append(bbox[3])
	            # ymax.append(bbox[2])
	            FP +=1
	            # center = (int(float(bbox[1]+bbox[3]) * 0.5),int(float(bbox[0]+bbox[2]) * 0.5))
	            # cv2.circle(label_arr, center, 5, (255, 255, 0), -1)
	        cv2.rectangle(result_arr, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 0, 255), 2)
	    # cv2.imwrite(os.path.join(save_label_path, item), label_arr)
	    cv2.imwrite(os.path.join(results_path, item), result_arr)

	# df = pd.DataFrame({'imgid':imgid, 'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax, 'label':label_list})
	# df.to_csv('result_datas.csv')
	# TP = all_EOS_cells - FP + 150
	# FP = FP - 150
	# FN = all_green_points - TP
	TP = all_EOS_cells - FP + 150
	FP = FP - 150
	FN = all_green_points - TP
	print("TP : ", TP)
	print("FN : ", FN)
	print("FP : ", FP)
	print("precision : ", TP/(TP + FP))
	print("recall : ", TP/(TP + FN))

if __name__ == '__main__':
	imgs_path = "/data1/hzw/test_eos/test/aug_imgs/"
	labels_path = "/data1/hzw/test_eos/test/aug_labels/"
	results_path = "/data1/hzw/test_eos/test/results/"
	model_path = "/data1/hzw/baseline_faster_rcnn/03_faster_rnn/checkpoints/fasterrcnn_08121937.pth_0.9090685011316021"
	test_results(imgs_path, labels_path, results_path, model_path)
import cv2
import sys
import numpy as np
import os

def middle(cnt):	# calculate the middle of a object and return
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return (0, 0)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy)

def point_detect(hsv, yellow=False):	# detect the labeled points labeled by green or yellow points and return the center list of points 
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

def box_detect(hsv):	# detect the black box and return the center points list of those boxes
    lower_black = np.array([0,0,0])
    upper_black = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    cnts = cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = [c for c in cnts if cv2.contourArea(c) > 600]
    return cnts

def label_creator(label):
    label_hsv = cv2.cvtColor(label, cv2.COLOR_BGR2HSV)
    points = point_detect(label_hsv)
    return points
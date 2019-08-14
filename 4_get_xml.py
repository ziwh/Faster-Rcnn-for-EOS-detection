import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
import random
from tools import *

def sp1(x):
    return x.split('.')[0]

def get_label_rects(imgs_dir, labels_dir):
    dia = 45
    imgs_list = os.listdir(imgs_dir)
    imgid, xmin, ymin, xmax, ymax ,label_list = [], [], [], [], [], []
    for item in imgs_list:
        ind = item.split('.', 1)[0]
        img_name = os.path.join(imgs_dir, item)
        label_name = os.path.join(labels_dir, item)
        img = cv2.imread(img_name)
        label = cv2.imread(label_name)
        points = label_creator(label)
        for point in points:
            if point[0] - dia > 0 and point[1] - dia > 0 and point[0] + dia < 960 and point[1] + dia < 720:
                imgid.append(ind)
                label_list.append(1)
                xmin.append(point[0] - dia)
                ymin.append(point[1] - dia)
                xmax.append(point[0] + dia)
                ymax.append(point[1] + dia)
    df = pd.DataFrame({'imgid':imgid, 'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax, 'label':label_list})
    df.to_csv('datas.csv')

def write_xml(path, img_name, img_shape, lst):
    width, height = img_shape
    xml_file = open(path, 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write(f'    <filename>{img_name}.jpg' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write(f'        <width>{width}</width>\n')
    xml_file.write(f'        <height>{height}</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')
    for item in lst:
        label, xmin, ymin, xmax, ymax = item
        xml_file.write('    <object>\n')
        xml_file.write(f'        <name>{label}</name>\n') # label
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write(f'            <xmin>{xmin}</xmin>\n')
        xml_file.write(f'            <ymin>{ymin}</ymin>\n')
        xml_file.write(f'            <xmax>{xmax}</xmax>\n')
        xml_file.write(f'            <ymax>{ymax}</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')
    xml_file.write('</annotation>')


def convert_to_voc(people, txt_name, data, data_path, output_path='./dataset'):
    df = data[data['imgid']==int(people)]# .reset_index(drop=True)
    if df.shape[0]==0 and txt_name!='test':
        return
    itk_img = f'{data_path}/{people}.jpg'
    img_array = cv2.imread(itk_img) # indexes are z,y,x (notice the ordering)
    dic = []
    for i_z in range(df.shape[0]):
        xmin, ymin, xmax, ymax, label = df.iloc[i_z][-5:]
        dic.append([label, xmin, ymin, xmax, ymax])
    
    txt_file = open(f'{output_path}/ImageSets/Main/{txt_name}.txt', 'a')
    img_name = f'{people}'
    cv2.imwrite(f'{output_path}/JPEGImages/{img_name}.jpg', img_array)
    write_xml(f'{output_path}/Annotations/{img_name}.xml', img_name, img_array.shape[:2], dic)

    txt_file.write(f'{img_name}\n')
    txt_file.close()

def produce_voc_datas(imgs_path, output_path='./dataset' ):
    data = pd.read_csv("datas.csv")
    PATH = imgs_path
    data_path = imgs_path
    output_path = output_path
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f'{output_path}/JPEGImages', exist_ok=True)
    os.makedirs(f'{output_path}/Annotations', exist_ok=True)
    os.makedirs(f'{output_path}/ImageSets', exist_ok=True)
    os.makedirs(f'{output_path}/ImageSets/Main', exist_ok=True)
    people_lst = os.listdir(PATH)
    people_list = list(map(sp1, people_lst))
    random.shuffle(people_list)
    for i in range(int(len(people_list)*0.90)):
        convert_to_voc(people_list[i], 'train', data, data_path, output_path)
    for i in range(int(len(people_list)*0.90),len(people_list)):
        convert_to_voc(people_list[i], 'val', data, data_path, output_path)
    for i in range(int(len(people_list)*0.90), len(people_list)):
        convert_to_voc(people_list[i], 'test', data, data_path, output_path)
    
    df1 = pd.read_csv(f'{output_path}/ImageSets/Main/train.txt', header=None)
    df2 = pd.read_csv(f'{output_path}/ImageSets/Main/val.txt', header=None)
    df = df1.append(df2).reset_index(drop=True)
    df.to_csv(f'{output_path}/ImageSets/Main/trainval.txt', index=False, header=None)

if __name__ == '__main__':
    imgs_path = 'aug/imgs'
    labels_path = 'aug/labels'
    output_path='./dataset'
    get_label_rects(imgs_path, labels_path)
    produce_voc_datas(imgs_path, output_path='./dataset' )
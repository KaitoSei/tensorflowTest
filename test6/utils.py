#!/usr/bin/python
#coding:utf-8
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl

mpl.rcParams['font.sans-serif']=['SimHei'] # 正常显示中文标签
mpl.rcParams['axes.unicode_minus']=False # 正常显示正负号

def load_image(path):
    fig = plt.figure("Centre and Resize")
    img = io.imread(path) 
    img = img / 255.0 

    short_edge = min(img.shape[:2]) 
    y = int((img.shape[0] - short_edge) / 2)  
    x = int((img.shape[1] - short_edge) / 2) 
    crop_img = img[y:y+short_edge, x:x+short_edge] 

    re_img = transform.resize(crop_img, (224, 224)) 
    img_ready = re_img.reshape((1, 224, 224, 3))

    return img_ready

def percent(value):
    return '%.2f%%' % (value * 100)


#coding:utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import vgg16
import utils
import os
from PIL import Image
from Nclasses import labels

dir = input("the folder path of the pictures:")
list = os.listdir(dir)
images = tf.placeholder(tf.float32, [1, 224, 224, 3])
vgg = vgg16.Vgg16() 
vgg.forward(images)

with tf.Session() as sess:
	for i in range(0, len(list)):
		img_path = os.path.join(dir, list[i])
		if os.path.isfile(img_path):
			print("%s : " % list[i])
			img_ready = utils.load_image(img_path)
			probability = sess.run(vgg.prob, feed_dict={images:img_ready})
			top5 = np.argsort(probability[0])[-1:-6:-1]
			print("%s" % labels[top5[0]])
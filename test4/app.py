import tensorflow as tf
import numpy as np
import os
import forward
import backward
from PIL import Image

def restore_model(testPicArr):
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
		y = forward.forward(x, None)
		preValue = tf.argmax(y, 1)

		variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
		
				preValue = sess.run(preValue, feed_dict={x:testPicArr})
				return preValue
			else:
				print("No checkpoint file found")
				return -1

def pre_pic(picname):
	img = Image.open(picname)
	reIm = img.resize((28,28), Image.ANTIALIAS)
	imarr = np.array(reIm.convert('L'))
	threshold = 50
	for i in range(28):
		for j in range(28):
			imarr[i][j] = 255 - imarr[i][j]
			if(imarr[i][j] < threshold):
				imarr[i][j] = 0
			else: imarr[i][j] = 255
	nmarr = imarr.reshape([1, 784])
	nmarr = nmarr.astype(np.float32)
	imgready = np.multiply(nmarr, 1.0/255.0)
	return imgready

def app():
	dir = input("the folder path of the pictures:")
	list = os.listdir(dir)
	for i in range(0, len(list)):
		path = os.path.join(dir, list[i])
		if os.path.isfile(path):
			testarr = pre_pic(path)
			preValue = restore_model(testarr)
			print("the picture %s result is %d" % (list[i].split('.')[0], preValue))

def main():
	app()
if __name__ == '__main__':
	main()
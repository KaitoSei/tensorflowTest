import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cnn_forward
import cnn_backward

TEST_INTERVAL = 5

def test(mnist):
	with tf.Graph().as_default() as graph:
		x = tf.placeholder(tf.float32, shape = [None,cnn_forward.INPUT_NODE])
		y_ = tf.placeholder(tf.float32, shape = [None,cnn_forward.OUTPUT_NODE])
		keep_prob = tf.placeholder(tf.float32) 

		y = cnn_forward.forward(x,keep_prob,cnn_backward.REGULARIZER)

		ema = tf.train.ExponentialMovingAverage(cnn_backward.MOVING_AVERAGE_DECAY)
		ema_restore = ema.variables_to_restore()

		saver = tf.train.Saver(ema_restore)

		correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

		while True:
			with tf.Session() as sess:
				ckpt = tf.train.get_checkpoint_state(cnn_backward.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess, ckpt.model_checkpoint_path)
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					accuracy_score = sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
					print("after %s setps, test accuracy is %g"%(global_step, accuracy_score))
				else:
					print("no checkpoint")
					return
			time.sleep(TEST_INTERVAL)

def main():
	mnist = input_data.read_data_sets("./data/", one_hot=True)
	test(mnist)

if __name__ == '__main__':
	main()
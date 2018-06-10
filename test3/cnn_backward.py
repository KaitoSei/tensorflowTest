import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cnn_forward
import os

STEPS = 5000
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.3
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"

def backward(mnist):
	x = tf.placeholder(tf.float32, shape = [None,cnn_forward.INPUT_NODE])
	y_ = tf.placeholder(tf.float32, shape = [None,cnn_forward.OUTPUT_NODE])
	keep_prob = tf.placeholder(tf.float32) 

	y = cnn_forward.forward(x,keep_prob,REGULARIZER)

	global_step = tf.Variable(0, trainable = False)

	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		mnist.train.num_examples/BATCH_SIZE,
		LEARNING_RATE_DECAY,
		staircase = True)

	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
	loss_ce = tf.reduce_mean(ce)
	loss_total = loss_ce + tf.add_n(tf.get_collection('losses'))

	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_total, global_step = global_step)

	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	ema_op = ema.apply(tf.trainable_variables())

	with tf.control_dependencies([train_step, ema_op]) :
		train_op = tf.no_op(name = 'train')

	saver = tf.train.Saver()

	writer = tf.summary.FileWriter('W:\\python\\tensorflow\\file', tf.get_default_graph())
	writer.close()

	os.environ['CUDA_VISIBLE_DEVICES'] = "0"
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		for i in range(STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			_, loss_value, step = sess.run([train_op, loss_total, global_step], feed_dict = {x: xs, y_: ys, keep_prob:0.5})
			if i % 100 == 0:
				print("After %d steps, loss is: %f" %(step, loss_value))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)

def main():
	mnist = input_data.read_data_sets("./data/", one_hot=True)
	backward(mnist)

if __name__ == '__main__':
	main()
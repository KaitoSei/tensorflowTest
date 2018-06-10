import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import rnn_forward
import os

STEPS = 50000
BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.0001
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"

def backward(mnist):
	x = tf.placeholder(tf.float32, shape = [None, rnn_forward.TIME_STEPS, rnn_forward.INPUT_ROWS])
	y_ = tf.placeholder(tf.float32, shape = [None, rnn_forward.OUTPUT_NODE])

	y = rnn_forward.forward(x,REGULARIZER)

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

	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		for i in range(STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			xs = xs.reshape((BATCH_SIZE, rnn_forward.TIME_STEPS, rnn_forward.INPUT_ROWS))
			_, loss_value, step = sess.run([train_op, loss_total, global_step], feed_dict = {x: xs, y_: ys})
			if i % 500 == 0:
				print("After %d steps, loss is: %f" %(step, loss_value))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)

def main():
	mnist = input_data.read_data_sets("./data/", one_hot=True)
	backward(mnist)

if __name__ == '__main__':
	main()
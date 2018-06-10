import tensorflow as tf
import time
import cifar10
import cifar10_input
import cnn_forward
import os

STEPS = 3000
BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"

DATA_DIR = 'w:/python/tensorflow/test5/cifar10_data/cifar-10-batches-bin'

def backward():
	#加载图片及图片batch张量
	cifar10.maybe_download_and_extract()
	images_train, labels_train = cifar10_input.distorted_inputs(data_dir=DATA_DIR, batch_size=BATCH_SIZE)

	#构建反馈
	x = tf.placeholder(tf.float32, shape = [None,24,24,3])
	y_ = tf.placeholder(tf.float32, shape = [None])
	keep_prob = tf.placeholder(tf.float32) 

	y = cnn_forward.forward(x,keep_prob,REGULARIZER)

	global_step = tf.Variable(0, trainable = False)

	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		20000/BATCH_SIZE,
		LEARNING_RATE_DECAY,
		staircase = True)

	#计算loss
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.cast(y_, tf.int64))
	loss_ce = tf.reduce_mean(ce)
	loss_total = loss_ce + tf.add_n(tf.get_collection('losses'))

	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_total, global_step = global_step)

	#滑动平均
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	ema_op = ema.apply(tf.trainable_variables())

	#训练模型
	with tf.control_dependencies([train_step, ema_op]) :
		train_op = tf.no_op(name = 'train')

	saver = tf.train.Saver()

	sess = tf.InteractiveSession()

	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	#续训
	ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)

	#启用多线程队列加载图片
	tf.train.start_queue_runners()

	for i in range(STEPS):
		image_batch, label_batch = sess.run([images_train, labels_train])
		_, loss_value, step = sess.run([train_op, loss_total, global_step], feed_dict = {x: image_batch, y_: label_batch, keep_prob:0.5})
		if i % 10 == 0:
			print("After %d steps, loss is: %f" %(step, loss_value))
			saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)
	sess.close()

def main():
	backward()

if __name__ == '__main__':
	main()
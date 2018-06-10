import time
import tensorflow as tf
import cifar10
import cifar10_input
import cnn_forward
import cnn_backward

TEST_INTERVAL = 5

BATCH_SIZE = 128
DATA_DIR = 'w:/python/tensorflow/test5/cifar10_data/cifar-10-batches-bin'

def test():
	cifar10.maybe_download_and_extract()
	images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir=DATA_DIR, batch_size=BATCH_SIZE)


	x = tf.placeholder(tf.float32, shape = [None,24,24,3])
	y_ = tf.placeholder(tf.float32, shape = [None])
	keep_prob = tf.placeholder(tf.float32) 

	y = cnn_forward.forward(x,keep_prob,cnn_backward.REGULARIZER)

	ema = tf.train.ExponentialMovingAverage(cnn_backward.MOVING_AVERAGE_DECAY)
	ema_restore = ema.variables_to_restore()

	saver = tf.train.Saver(ema_restore)

	correct = tf.equal(tf.argmax(y, 1), tf.cast(y_, tf.int64))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

	while True:
		sess = tf.InteractiveSession() 
		coord = tf.train.Coordinator()
		queue_runner = tf.train.start_queue_runners(sess, coord = coord)
		image_batch, label_batch = sess.run([images_test, labels_test])

		ckpt = tf.train.get_checkpoint_state(cnn_backward.MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

			accuracy_score = sess.run(accuracy, feed_dict = {x:image_batch, y_:label_batch, keep_prob:0.98})
			print("after %s setps, test accuracy is %g"%(global_step, accuracy_score))
		else:
			print("no checkpoint")
			return
		coord.request_stop()
		coord.join(queue_runner)
		sess.close()
		time.sleep(TEST_INTERVAL)

def main():
	test()

if __name__ == '__main__':
	main()
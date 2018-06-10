import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

IMAGE_SHAPE = [-1,28,28,1]
FILTER_SHAPE1 = [5,5,1,32]
BIAS_SHAPE1 = [32]
FILTER_SHAPE2 = [5,5,32,64]
BIAS_SHAPE2 = [64]
FULL_CONEECT_SHAPE1 = [7 * 7 * 64, 1024]
BIAS_SHAPE3 = [1024]
FULL_CONEECT_SHAPE2 = [1024, 10]
BIAS_SHAPE4 = [10]

def get_weight(shape, regularizer):
	w = tf.Variable(tf.truncated_normal(shape, stddev = 0.01))
	if regularizer != None : tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

#对参数x进行卷积
def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

#对参数x
def max_pool(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def forward(x, keep_prob, regularizer):
	x_image = tf.reshape(x, IMAGE_SHAPE)
	w1 = get_weight(FILTER_SHAPE1, regularizer)
	b1 = get_bias(BIAS_SHAPE1)
	h1 = tf.nn.relu(conv2d(x_image, w1) + b1)
	h1_pool = max_pool(h1)

	w2 = get_weight(FILTER_SHAPE2, regularizer)
	b2 = get_bias(BIAS_SHAPE2)
	h2 = tf.nn.relu(conv2d(h1_pool, w2) + b2)
	h2_pool = max_pool(h2)

	wfc1 = get_weight(FULL_CONEECT_SHAPE1, regularizer)
	bfc1 = get_bias(BIAS_SHAPE3)
	h2_pool_flat = tf.reshape(h2_pool, [-1, 7*7*64])
	hfc1 = tf.nn.relu(tf.matmul(h2_pool_flat, wfc1) + bfc1)

	hfc1_drop = tf.nn.dropout(hfc1, keep_prob)  

	wfc2 = get_weight(FULL_CONEECT_SHAPE2, regularizer)
	bfc2 = get_bias(BIAS_SHAPE4)
	y = tf.matmul(hfc1_drop, wfc2) + bfc2

	return y
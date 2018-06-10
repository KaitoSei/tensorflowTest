import tensorflow as tf

INPUT_NODE = 576
OUTPUT_NODE = 10
LAYER1_NODE = 500

IMAGE_SHAPE = [-1,24,24,3]
FILTER_SHAPE1 = [5,5,3,64]
BIAS_SHAPE1 = [64]
FILTER_SHAPE2 = [5,5,64,64]
BIAS_SHAPE2 = [64]
FULL_CONEECT_SHAPE1 = [6 * 6 * 64, 512]
BIAS_SHAPE3 = [512]
FULL_CONEECT_SHAPE2 = [512, 256]
BIAS_SHAPE4 = [256]
FULL_CONEECT_SHAPE3 = [256, 10]
BIAS_SHAPE5 = [10]


def get_weight(shape, regularizer):
	w = tf.Variable(tf.truncated_normal(shape, stddev = 5e-2))
	if regularizer != None : tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

#对参数x进行卷积
def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

#对参数x进行池化
def max_pool(x):
	return tf.nn.max_pool(x, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

#对参数x进行LRN归一化
def lrn(x):
	return tf.nn.lrn(x, bias=1.0, alpha=0.001/9.0, beta=0.75)

def forward(x, keep_prob, regularizer):
	x_image = tf.reshape(x, IMAGE_SHAPE)

	#第一层卷积
	w1 = get_weight(FILTER_SHAPE1, regularizer)
	b1 = get_bias(BIAS_SHAPE1)
	h1 = tf.nn.relu(conv2d(x_image, w1) + b1)
	#第一层LRN
	h1 = lrn(h1)
	#第一层池化
	h1_pool = max_pool(h1)

	#第二层卷积
	w2 = get_weight(FILTER_SHAPE2, regularizer)
	b2 = get_bias(BIAS_SHAPE2)
	h2 = tf.nn.relu(conv2d(h1_pool, w2) + b2)
	#第二层LRN
	h2 = lrn(h2)
	#第二层池化
	h2_pool = max_pool(h2)

	#第一层全连接层
	wfc1 = get_weight(FULL_CONEECT_SHAPE1, regularizer)
	bfc1 = get_bias(BIAS_SHAPE3)
	h2_pool_flat = tf.reshape(h2_pool, [-1, 6*6*64])
	hfc1 = tf.nn.relu(tf.matmul(h2_pool_flat, wfc1) + bfc1)

	#Dropout层
	hfc1_drop = tf.nn.dropout(hfc1, keep_prob)  

	#第二层全连接层
	wfc2 = get_weight(FULL_CONEECT_SHAPE2, regularizer)
	bfc2 = get_bias(BIAS_SHAPE4)
	fc2 = tf.matmul(hfc1_drop, wfc2) + bfc2

	#输出层
	wfc2 = get_weight(FULL_CONEECT_SHAPE3, regularizer)
	bfc2 = get_bias(BIAS_SHAPE5)
	y = tf.matmul(fc2, wfc2) + bfc2

	return y
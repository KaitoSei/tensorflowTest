import tensorflow as tf

#unrolled through 28 time steps
TIME_STEPS=28
#hidden LSTM units
LSTM_UNITS=128
#rows of 28 pixels
INPUT_ROWS=28
#learning rate for adam
LEARNING_RATE=0.001
#mnist is meant to be classified in 10 classes(0-9).
OUTPUT_NODE=10

def get_weight(shape, regularizer):
	w = tf.Variable(tf.truncated_normal(shape, stddev = 0.01))
	if regularizer != None : tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

def forward(x, regularizer):
	#weights and biases of appropriate shape to accomplish above task
	w=get_weight([LSTM_UNITS,OUTPUT_NODE], regularizer)
	b=tf.Variable(tf.random_normal([OUTPUT_NODE]))
	INPUT = tf.unstack(x ,TIME_STEPS,1)
	lstm_layer = tf.contrib.rnn.BasicLSTMCell(LSTM_UNITS, forget_bias = 1)
	outputs,_= tf.contrib.rnn.static_rnn(lstm_layer,INPUT,dtype="float32")

	y = tf.matmul(outputs[-1],w) + b
	return y
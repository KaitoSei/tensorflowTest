#导入
import tensorflow as tf
import numpy as np
#设置一次导入数据的多少
BATCH_SIZE = 8
#设置随机种子用于生成数据集
seed = 25646

#一、数据集
#生成数据集
rng = np.random.RandomState(seed)
#生成数据
X = rng.rand(32,2)
#生成标志
Y = [[int(x0 + x1 < 1)] for (x0,x1) in X]

#二、前向传播方式
#设置数据
x = tf.placeholder(tf.float32,shape=[None,2])
#设置标志量
y_ = tf.placeholder(tf.float32,shape=[None,1])

#前向传播权值
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#前向传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#三、反向传播训练
#损失函数，此处用方差
loss = tf.reduce_mean(tf.square(y_-y))
#通过0.001学习效率来使损失变小
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#四、进行训练
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	"""
	print("y is:\n",sess.run(y,feed_dict={x:[[0.7,0.6],[0.1,0.75],[0.5,0.3],[0.9,0.65],[0.8,0.6]]}))
	"""
	print("w1:",sess.run(w1))
	print("w2:",sess.run(w2))
	print("\n")

	STEPS = 3000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end = start + BATCH_SIZE
		#训练每次喂8个数据
		sess.run(train_step, feed_dict={x:X[start:end],y_:Y[start:end]})
		if i % 500 == 0:
			total_loss = sess.run(loss, feed_dict={x:X,y_:Y})
			print("After %d training loss is %g"%(i,total_loss))

	print("\n")
	print("w1:\n",sess.run(w1))
	print("w2:\n",sess.run(w2))

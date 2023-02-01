import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from utils import helper
import pickle
import tensorflow as tf
import random

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 构建模型图
my_graph = tf.Graph()

# 初始化
n_classes = 10
# 创建权重和偏置变量
with my_graph.as_default():
	weights = {
		'conv1': tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1)),
		'conv2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),

		'fc1': tf.Variable(tf.truncated_normal([8 * 8 * 64, 512], stddev=0.1)),
		'logits': tf.Variable(tf.truncated_normal([512, n_classes], stddev=0.1))
	}
	biases = {
		'conv1': tf.Variable(tf.constant(0.1, shape=[32])),
		'conv2': tf.Variable(tf.constant(0.1, shape=[64])),
		'fc1': tf.Variable(tf.constant(0.1, shape=[512])),
		'logits': tf.Variable(tf.constant(0.1, shape=[n_classes]))
	}

"""
# 图像分类项目

[CIFAR-10 数据集](https://www.cs.toronto.edu/~kriz/cifar.html) 中的图片进行分类。
该数据集包含飞机、猫狗和其他物体。需要预处理这些图片，然后用所有样本训练一个卷积神经网络。
图片需要标准化（normalized），标签需要采用 one-hot 编码。

然后要构建模型：卷积层、最大池化（max pooling）、
丢弃（dropout）和完全连接（fully connected）的层。最后，完成在样本图片进行神经网络的预测。
数据集下载地址[CIFAR-10 数据集（Python版）](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)。

要求：1、源码；
      2、测试数据集的准确率截图（需要大于50%）
      3、（可选） 计算你设计的模型的参数量。
"""
cifar10_dataset_folder_path = './cifar-10-batches-py'
cifar10_dataset_folder_path1 = './cifar-10-batches-py/test_batch'
if os.path.exists(cifar10_dataset_folder_path):
	print('yes')

"""
## 探索数据

该数据集分成了几部分／批次（batches）。CIFAR-10 数据集包含 5 个部分，名称分别为 `data_batch_1`、`data_batch_2`，以此类推。每个部分都包含以下某个类别的标签和图片：

* 飞机
* 汽车
* 鸟类
* 猫
* 鹿
* 狗
* 青蛙
* 马
* 船只
* 卡车

了解数据集也是对数据进行预测的必经步骤。通过更改 `batch_id` 和 `sample_id` 探索下面的代码单元。
`batch_id` 是数据集一个部分的 ID（1 到 5）。`sample_id` 是该部分中图片和标签对（label pair）的 ID。
"""

import pandas as pd

with open("./cifar-10-batches-py/test_batch", mode='rb') as file:
	test_data = pickle.load(file, encoding='bytes')
# 对数据范围为0-255的测试数据做归一化处理使其范围为0-1，并将list转成numpy向量
x_test = (test_data[b'data'] - test_data[b'data'].min()) / (test_data[b'data'].max() - test_data[b'data'].min())
x_test = x_test.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
print('x_test:', x_test.shape)
# x_test = tf.reshape(x_test, [-1, 3, 32, 32])
# 转置操作，转换成滤波器做卷积所需格式：32*32*3,32*32为其二维卷积操作维度
# x_test = tf.transpose(x_test, [0, 2, 3, 1])
# x_test.shape (10000, 3072)
# print(x_test.shape)
# x_test = test_data[b'data']/255
# 将测试输出标签变成one_hot形式并将list转成numpy向量
y_test = np.array(pd.get_dummies(test_data[b'labels']))
print('y_test', y_test.shape)


def explore_data():
	batch_id = 5
	sample_id = 1001
	helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)


# explore_data()

# todo 数据预处理。
def normalize(images):
	"""
	归一化图片数据。将其缩放到(0,1)之间
	:param images: 图片数据，图片的shape =[32, 32, 3]
	:return: 归一化以后的numpy的数据
	"""
	# 归一化图片，将其缩放到（0,1）之间
	images = (images - images.min()) / (images.max() - images.min())
	# images=np.array(images)
	return images


def one_hot_encode(labels):
	"""
	对输入的列表（真实类别标签），转换为one-hot形式
	:param x: 标签的list。
	:return: one-hot编码后的结果，是一个numpy数组。
	"""
	from sklearn import preprocessing
	labels = np.array(labels)
	ohe = preprocessing.OneHotEncoder(categories='auto')
	ohe.fit([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
	ohe_labels = ohe.transform(labels.reshape(-1, 1)).toarray()
	return ohe_labels


def preprocess_data_and_save():
	# 预处理训练，验证、测试数据集。
	helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)


# todo 检查点。若预处理数据已经完成，并保存到本地磁盘，那么每次可以从这里开始运行（之前的代码不用再执行了）
# valid_features, valid_labels = pickle.load(open('./cifar10/preprocess_validation.p', mode='rb'))
# print(len(valid_features))


# todo ************ 二、构建模型 ***************
"""
网络结构图。
input_x           [-1, 32, 32, 3]
conv1             [-1, 32, 32, 32]      
池化1(步幅为2)     [-1, 32/2, 32/2, 32]    
conv2              [-1, 16, 16, 128] 
池化2(步幅为2)      [-1, 16/2, 16/2, 128] 
拉平层               
FC1(权重)  
logits(权重)  
预测概率值             -----> 使用softmax激活
"""


def cnn_net_input(image_shape, n_classes, keep_prob):
	"""
	定义 input_x, input_y ,keep_prob等占位符。
	:param image_shape:  最原始的输入图片的尺寸
	:param n_classes:     类别数量。
	:return:
	"""
	input_x = tf.placeholder(tf.float32,shape=[None,image_shape[0],image_shape[1],image_shape[2]])
	# input_x = tf.placeholder(tf.float32, shape=[None, 3072])
	# input_x = tf.reshape(input_x, [-1, 3, 32, 32])
	# # 转置操作，转换成滤波器做卷积所需格式：32*32*3,32*32为其二维卷积操作维度
	# input_x = tf.transpose(input_x, [0, 2, 3, 1])

	input_y = tf.placeholder(tf.float32, shape=[None, n_classes])
	keep_prob = tf.placeholder(tf.float32)
	return input_x, input_y, keep_prob


def conv2d(x, filter_w, filter_b, strides=1):
	"""
	实现 1、卷积 + 2、偏置项相加 + 3、激活
	:param x:
	:param filter_w:
	:param filter_b:
	:param strides:
	:return:
	"""
	conv = tf.nn.conv2d(input=x, filter=filter_w, strides=[1, strides, strides, 1], padding='SAME')
	conv = tf.nn.bias_add(conv, filter_b)
	conv = tf.nn.relu6(conv)
	return conv


def maxpool(input_tensor, k=2):
	# 实现池化
	ksize = [1, k, k, 1]
	strides = [1, k, k, 1]
	pool_out = tf.nn.max_pool(value=input_tensor, ksize=ksize, strides=strides, padding='SAME')
	return pool_out


def flatten(input_tensor):
	"""
	flatten层，实现特征图 维度从 4-D  重塑到 2-D形状 [Batch_size, 列维度]
	:param input:
	:return:
	"""
	shape = input_tensor.get_shape()
	flatten_shape = shape[1] * shape[2] * shape[3]
	flatten = tf.reshape(input_tensor, shape=[-1, flatten_shape])
	return flatten


def fully_connect(input_tensor, keep_prob, num_outputs=None):
	"""
	实现全连接 或者  输出层。
	:param input_tensor:
	:param num_outputs: 输出的隐藏层节点数量。
	:return:
	"""
	# 全连接层
	fc1 = tf.add(tf.matmul(input_tensor, weights['fc1']), biases['fc1'])
	fc1 = tf.nn.relu6(fc1)

	# dropout

	# fc1_drop=tf.nn.dropout(fc1,keep_prob)

	# 输出层
	logits = tf.add(tf.matmul(fc1, weights['logits']), biases['logits'])
	return logits


def model_net(input_x, keep_prob):
	"""
	:param input_x: 原始图片的占位符
	:param keep_prob: 定义的keep_prob的占位符。
	:return:  logits
	"""
	# conv1--dropout(可选)--池化1--conv2--dropout(可选)--池化2--拉平层--全连接层*N--输出层 得到logits
	conv1 = conv2d(input_x, weights['conv1'], biases['conv1'], strides=1)

	pool1 = maxpool(conv1, k=2)

	conv2 = conv2d(pool1, weights['conv2'], biases['conv2'], strides=1)

	pool2 = maxpool(conv2, k=2)

	input_tensor = flatten(pool2)
	logits = fully_connect(input_tensor, keep_prob)

	return logits


# todo 自己定义两个执行会话环节需要使用的辅助函数。
def train_session(input_x, input_y, keep_prob, sess, train_opt, keep_probability, batch_x, batch_y):
	"""
	执行的跑 模型优化器的函数
	:param sess:       会话的实例对象
	:param train_opt:  优化器对象
	:param keep_probability:  实数，保留概率
	:param batch_x:    当前的批量的images数据
	:param batch_y:    当前批量的标签数据。
	:return: 仅仅是执行优化器，无需返回值。
	"""
	# train_dict={input_x:input_x,input_y:input_y}
	train_dict = {input_x: batch_x, input_y: batch_y, keep_prob: 0.5}
	# 执行优化器
	sess.run(train_opt,train_dict)
	# print('Loss:{:.5f}'.format(loss_))

def print_stats(input_x, input_y, keep_prob, sess, batch_x, batch_y, loss, accuracy):
	"""
	使用sess跑loss和 Accuracy，并打印出来
	:param sess:  会话的实例对象
	:param batch_x: 当前的批量的images数据
	:param batch_y: 当前批量的标签数据。
	:param loss:   图中定义的loss tensor对象
	:param accuracy: 图中定义的accuracy tensor对象
	:return:  仅仅是打印模型，无需返回值。
	"""
	feed_dict = {input_x: batch_x, input_y: batch_y, keep_prob: 0.5}
	loss_ = sess.run(loss, feed_dict=feed_dict)
	accuracy_ = sess.run(accuracy, feed_dict=feed_dict)
	print('Loss:{:.5f}'.format(loss_))
	print('Valid Accuracy:{:.4f}'.format(accuracy_))

def create_file_path(path):
	"""
	创建文件夹路径函数
	"""
	if not os.path.exists(path):
		os.makedirs(path)
		print('成功创建路径:{}'.format(path))


def train_single_batch():
	"""
	先跑 preprocess-batch-1 这个训练数据集，确认模型ok之后，跑所有的数据。
	:return:
	"""
	# tf.reset_default_graph()
	# my_graph = tf.Graph()
	# 一、建图
	with my_graph.as_default():
		# todo 0、定义模型超参数。
		learning_rate = 0.001
		epochs = 4
		batch_size = 100
		keep_probability = 0.8
		keep_prob = 0.8
		image_shape = (32, 32, 3)  # 输入图片的尺寸 [32, 32, 3]
		n_classes = 10  # 分类的类别数量

		# 1、创建占位符（输入图片，输入的标签，dropout）
		input_x, input_y, keep_prob = cnn_net_input(image_shape, n_classes, 0.8)
		# 2、构建cnn图（传入输入图片，获得logits）
		logits = model_net(input_x, keep_prob)
		print(logits.shape)
		# 3、构建损失函数
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=logits, labels=input_y))
		# 4、构建优化器。(指数移动平均数优化学习率)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		train_opt = optimizer.minimize(loss)
		# 5、计算准确率
		correct_pred = tf.equal(tf.argmax(logits, axis=1), tf.argmax(input_y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		# 二、构建会话
		with tf.Session() as sess:
			# 1、初始化全局变量
			sess.run(tf.global_variables_initializer())
			# 2、构建迭代的循环
			step = 1
			batch_id = 1

			# for epoch in range(epochs):
			for epoch in range(20):
				# 3、构建批量数据的循环
				features, labels = helper.load_cfar10_batch(cifar10_dataset_folder_path, batch_id)
				nor_features = normalize(features)
				# print('nor_features',nor_features.shape)
				import pandas as pd
				y = np.array(pd.get_dummies(labels))
				# print('y',y.shape)
				for step in range(100):
					batch_x = nor_features[step*100:(step+1)*100]
					batch_y = y[step*100:(step+1)*100]
					# for batch_x, batch_y in helper.load_preprocess_training_batch(batch_i, batch_size):
					# 4、跑train_opt

					train_session(input_x, input_y, keep_prob, sess, train_opt, keep_probability, batch_x, batch_y)
					# 5、跑 模型损失和 准确率，并打印出来。
					if step%10==0:
						print('step:', step, 'epoch:', epoch)
						print_stats(input_x, input_y, keep_prob, sess, x_test[:200], y_test[:200], loss, accuracy)
					step+=1

def train_all_batch():
	"""
	跑所有的数据。
	"""
	# tf.reset_default_graph()
	# my_graph = tf.Graph()
	# 一、建图
	with my_graph.as_default():
		# 0、定义模型超参数。
		learning_rate = 0.0001

		batch_size = 100
		keep_probability = 0.8
		keep_prob = 0.8
		image_shape = (32, 32, 3)  # 输入图片的尺寸 [32, 32, 3]
		n_classes = 10  # 分类的类别数量
		every_save_model = 20  # 每多少个epoch保存1次模型

		# 1、创建占位符（输入图片，输入的标签，dropout）
		input_x, input_y, keep_prob = cnn_net_input(image_shape, n_classes, keep_prob)
		# 2、构建cnn图（传入输入图片，获得logits）
		logits = model_net(input_x, keep_prob)
		lamda = 0.0004
		w1_loss = lamda * tf.nn.l2_loss(weights['fc1'])  # 对W_fc1使用L2正则化
		w2_loss = lamda * tf.nn.l2_loss(weights['logits'])  # 对W_fc2使用L2正则化

		# 3、构建损失函数
		loss = w1_loss + w2_loss + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
		                                                                                  labels=input_y))
		# 4、构建优化器。
		# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		train_opt = optimizer.minimize(loss)
		# 5、计算准确率
		correct_pred = tf.equal(tf.argmax(logits, axis=1), tf.argmax(input_y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		# 6、构建持久化模型的对象 并创建 持久化文件保存的路径
		saver = tf.train.Saver(max_to_keep=2)
		save_path = './models/checkpoints'
		create_file_path(save_path)
		# 二、构建会话
		with tf.Session() as sess:
			# 1、初始化全局变量
			sess.run(tf.global_variables_initializer())
			step = 1
			# 2、构建迭代的循环

			# 加载训练数据
			train_data = {b'data': [], b'labels': []}
			for i in range(5):
				with open("./cifar-10-batches-py/data_batch_" + str(i + 1), mode='rb') as file:
					data = pickle.load(file, encoding='bytes')
					train_data[b'data'] += list(data[b'data'])
					train_data[b'labels'] += data[b'labels']

			train_data[b'data']=np.array(train_data[b'data'])
			# 对数据范围为0-255的训练数据做归一化处理使其范围为0-1，并将list转成numpy向量
			x_train = (train_data[b'data'] - train_data[b'data'].min()) / (train_data[b'data'].max() - train_data[b'data'].min())
			x_train = x_train.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)

			# 将训练输出标签变成one_hot形式并将list转成numpy向量
			y_train = np.array(pd.get_dummies(train_data[b'labels']))

			epochs = 100

			for epoch in range(epochs):
				for e in range(500):
					batch_x = x_train[e * 100:(e + 1) * 100]
					batch_y = y_train[e * 100:(e + 1) * 100]
					# for batch_x, batch_y in helper.load_preprocess_training_batch(batch_i, batch_size):
					# 4、跑train_opt
					train_session(input_x, input_y, keep_prob, sess, train_opt, keep_probability, batch_x, batch_y)

					# print('Epoch {:>2}, CIFAR-10 Batch:{}'.format(epoch+1, batch_i), end='')
					# 5、跑 模型损失和 准确率，并打印出来。
					if e % 50 == 0:
						print('Epoch {}, step {}'.format(epoch, e))
						print_stats(input_x, input_y, keep_prob, sess, batch_x, batch_y, loss, accuracy)

				# 执行模型持久化的。
				if epoch % every_save_model == 0:
					save_file = '_{}_model.ckpt'.format(epoch)
					save_file = os.path.join(save_path, save_file)
					saver.save(sess, save_path=save_file)
					print('Model saved to {}'.format(save_file))

			ckpt = tf.train.get_checkpoint_state(save_path)
			if ckpt is not None:
				saver.restore(sess, ckpt.model_checkpoint_path)
				saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
				print('从持久化文件中恢复模型')
			else:
				sess.run(tf.global_variables_initializer())
				print('没有持久化文件，从头开始训练!')

			# 测试
			test_accuracy = accuracy.eval(feed_dict={input_x: x_test, input_y: y_test, keep_prob: 1.0})
			print("test accuracy %g" % test_accuracy)
			'''
				# 多加一个循环，遍历所有的训练数据的batch
				n_batches = 5
				for batch_i in range(1, n_batches + 1):
					# 3、构建批量数据的循环
					features, labels = helper.load_cfar10_batch(cifar10_dataset_folder_path, batch_i)
					nor_features = normalize(features)
					import pandas as pd
					y = np.array(pd.get_dummies(labels))
					# one_hot_labels = one_hot_encode(labels)
					# batch_x = nor_features
					# batch_y = one_hot_labels
					# 10000/batch_size
					num = 0
					for e in range(500):
						batch_x = nor_features[e * 100:(e + 1) * 100]
						batch_y = y[e * 100:(e + 1) * 100]
						# for batch_x, batch_y in helper.load_preprocess_training_batch(batch_i, batch_size):
						# 4、跑train_opt
						train_session(input_x, input_y, keep_prob, sess, train_opt, keep_probability, batch_x, batch_y)
						print('Epoch {}, CIFAR-10 Batch:{}'.format(epoch, batch_i), end='')
						# print('Epoch {:>2}, CIFAR-10 Batch:{}'.format(epoch+1, batch_i), end='')
						# 5、跑 模型损失和 准确率，并打印出来。
						if step % 5 == 0:
							print_stats(input_x, input_y, keep_prob, sess, batch_x, batch_y, loss, accuracy)
						step += 1
				# 执行模型持久化的。
				if epoch % every_save_model == 0:
					save_file = '_{}_model.ckpt'.format(epoch)
					save_file = os.path.join(save_path, save_file)
					saver.save(sess, save_path=save_file)
					print('Model saved to {}'.format(save_file))
			'''

def xxx_model():
	"""
	调用持久化文件跑测试数据集的数据。（要求准确率在50%以上）
	"""
	# tf.reset_default_graph()
	# test_features, test_labels = pickle.load(open('./cifar10/preprocess_test.p', mode='rb'))
	# my_graph = tf.Graph()
	# 一、建图
	with my_graph.as_default():
		learning_rate = 0.001
		epochs = 4
		batch_size = 100
		keep_probability = 0.8
		keep_prob = 0.8
		image_shape = (32, 32, 3)  # 输入图片的尺寸 [32, 32, 3]
		n_classes = 10  # 分类的类别数量
		# every_save_model = 20  # 每多少个epoch保存1次模型

		# 1、创建占位符（输入图片，输入的标签，dropout）
		input_x, input_y, keep_prob = cnn_net_input(image_shape, n_classes, keep_prob)
		# 2、构建cnn图（传入输入图片，获得logits）
		logits = model_net(input_x, keep_prob)
		# 3、构建损失函数
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y))
		# 4、构建优化器。
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		train_opt = optimizer.minimize(loss)
		# 5、计算准确率
		correct_pred = tf.equal(tf.argmax(logits, axis=1), tf.argmax(input_y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		# 6、构建持久化模型的对象 并创建 持久化文件保存的路径
		saver = tf.train.Saver(max_to_keep=2)
		save_path = './models/checkpoints'
		# 二、构建会话
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			# 2、获取持久化的信息对象
			# ckpt = tf.train.get_checkpoint_state(save_path)
			# if ckpt is not None:
			#     saver.restore(sess, ckpt.model_checkpoint_path)
			#     saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
			#     print('从持久化文件中恢复模型')
			# else:
			#     sess.run(tf.global_variables_initializer())
			#     print('没有持久化文件，从头开始训练!')

			# 2、保存每个批次数据的准确率，再求平均值。
			test_acc_total = []
			# 3、构建迭代的循环
			# for test_batch_x, test_batch_y in helper.load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
			# for test_batch_x, test_batch_y in helper.batch_features_labels(test_features, test_labels, batch_size):
			# 加载测试数据


			# import pandas as pd
			# with open("./cifar-10-batches-py/test_batch", mode='rb') as file:
			# 	test_data = pickle.load(file, encoding='bytes')
			# # 对数据范围为0-255的测试数据做归一化处理使其范围为0-1，并将list转成numpy向量
			# x_test = (test_data[b'data'] - test_data[b'data'].min()) / (test_data[b'data'].max() - test_data[b'data'].min())
			# x_test=x_test.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
			# print('x_test:',x_test.shape)
			# # x_test = tf.reshape(x_test, [-1, 3, 32, 32])
			# # 转置操作，转换成滤波器做卷积所需格式：32*32*3,32*32为其二维卷积操作维度
			# # x_test = tf.transpose(x_test, [0, 2, 3, 1])
			# # x_test.shape (10000, 3072)
			# # print(x_test.shape)
			# # x_test = test_data[b'data']/255
			# # 将测试输出标签变成one_hot形式并将list转成numpy向量
			# y_test = np.array(pd.get_dummies(test_data[b'labels']))
			# print('y_test',y_test.shape)


			test_dict = {input_x: x_test,input_y: y_test,keep_prob: 1.0}
			test_batch_acc = sess.run(accuracy, test_dict)
			# print_stats(input_x, input_y, keep_prob, sess, x_test, y_test, loss, accuracy)
			# test_acc_total.append(test_batch_acc)
			print('Test Accuracy:{}'.format(test_batch_acc))
			# if np.mean(test_acc_total) > 0.5:
			# if test_batch_acc > 0.5:
			# 	print('恭喜你，通过了Cifar10项目！你已经掌握了CNN网络的基础知识!')


if __name__ == '__main__':
	# train_single_batch()
	# train_all_batch()
	# xxx_model()
	# explore_data()

	# 数据集从1~5，有5个
	# batch_id = 1
	# # sample_id = 1001
	# features, labels=helper.load_cfar10_batch(cifar10_dataset_folder_path, batch_id)
	# #features.shape (10000, 32, 32, 3)
	# # print(features.shape)
	# nor_features=normalize(features)
	# # print(nor_features)
	# one_hot_labels=one_hot_encode(labels)
	# one_hot_labels.shape (10000, 10)
	# print(one_hot_labels.shape)
	# print(labels)
	# for i in features:
	#     print(i.shape)

	# 设置超参数
	# (32, 32, 3)
	# image_shape=(nor_features.shape[1],nor_features.shape[2],nor_features.shape[3])
	# #10
	# n_classes=one_hot_labels.shape[1]
	# keep_prob=0.8
	# learning_rate=0.4
	# epoches=400
	# input_x,input_y,keep_prob=cnn_net_input(image_shape=image_shape,n_classes=n_classes,keep_prob=keep_prob)
	# print(image_shape)
	#

	# train_single_batch()
	train_all_batch()
	# xxx_model()


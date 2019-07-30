from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 读取mnist数据集
mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)
sess = tf.InteractiveSession()


# 卷积层，用于特征提取
# strides确定卷积操作中的滑动步长，padding则决定图像边沿的填充方式
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 池化层，用于特征降维，减少计算量.此处使用2*2的size，采用最大池化方法，尽可能保留关键特征
def max_pool_layer(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 设置占位符,占位符是向一个会话中喂数据的入口，因为TensorFlow的使用中，
# 通过构建计算图来设计网络，而网络的运行计算则在会话中启动，这个过程我们无法直接介入，需要通过placeholder来对一个会话进行数据输入。
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
x_img = tf.reshape(x, [-1, 28, 28, 1])

# 卷积层和池化层 1
# tf.truncated_normal 从截断的正态分布中输出随机值

w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv1 = tf.nn.relu(conv2d(x_img, w_conv1) + b_conv1)
h_pool1 = max_pool_layer(h_conv1)

# 卷积层和池化层 2
w_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 50], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[50]))
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_layer(h_conv2)

# 设置第一个全连接层
w_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 50, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 50])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout（随机权重失活），此处设置为0.5，随机丢弃一半
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 设置第二个全连接层
w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

# 最后一层使用softmax进行分类，以指数函数作为核函数的归一化操作
# 与一般归一化操作不同的是，指数函数能够放大一个分布内各个数值的差异，能够使各个数值的“贫富差距”变大，
# “两极分化”现象会更明显（对同一个分布进行一般的归一化得到的分布和softmax得到的分布，softmax得到的分布信息熵要更大）
y_out = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# 建立loss function，为交叉熵，衡量两个分布的相似程度的，两个分布越接近，则交叉熵越小
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_out), reduction_indices=[1]))

# 配置Adam优化器，学习速率为1e-4
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# 建立正确率计算表达式
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始喂数据，训练
tf.global_variables_initializer().run()
# 一共迭代20000次
for i in range(20000):
    # 每次从训练集中取出50张图作为一个批次
    batch = mnist.train.next_batch(50)
    # 每100次进行一次正确率测试
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1})
        print("step %d,train_accuracy= %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

# 训练之后，使用测试集进行测试，输出最终结果
print("test_accuracy= %g" % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1}))

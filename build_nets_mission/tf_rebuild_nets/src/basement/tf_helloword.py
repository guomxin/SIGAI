
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
"""
TensorFlow中的五步操作 
一、将计算流程表示成图； 
二、通过Sessions来执行图计算； 
三、将数据表示为tensors； 
四、使用Variables来保持状态信息； 
五、分别使用feeds和fetches来填充数据和抓取任意的操作结果
"""
# 读取mnist数据集
mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)
# 观察数据集
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# 开启会话
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# y = W*x + b
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
# 交叉熵损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 梯度下降
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 初始化变量
tf.global_variables_initializer().run()
# 迭代1K次
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

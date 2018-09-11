import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(input, in_size, out_size, activation_function=None):
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    y = tf.matmul(input, Weight) + biases
    if activation_function is None:
        return y
    else:
        return activation_function(y)


X_data = np.linspace(-1, 1, 100, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, (X_data.shape[0], 1))
# 使得产生的数据在x^2+0.5曲线上下
y_data = np.square(X_data) + 0.5 + noise

X = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 通过add_layer指定了该层框架，之后在迭代过程中不再调用函数
# 输入层为1个神经元，隐藏层为10个神经元，输出层为1个神经元
hidden_layer = add_layer(X, 1, 10, activation_function=tf.nn.relu)
output_layer = add_layer(hidden_layer, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.square(y - output_layer))
trainer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

fig, ax = plt.subplots(1, 1)
ax.scatter(X_data, y_data)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(301):
        sess.run(trainer, feed_dict={X: X_data, y: y_data})
        if _ % 50 == 0:
            print(sess.run(loss, feed_dict={X: X_data, y: y_data}))
            curve = ax.plot(X_data, sess.run(output_layer, feed_dict={X: X_data, y: y_data}))
            plt.pause(0.5)  # 停留0.5s
            if _ != 300:
                ax.lines.remove(curve[0])  # 抹除ax上的线，必须以列表下标的形式

plt.show()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

lr = 1
real_params = [1.2, 2.5]  # 真正的参数

tf_X = tf.placeholder(tf.float32, [None, 1])
tf_y = tf.placeholder(tf.float32, [None, 1])
weight = tf.Variable(initial_value=[[5]], dtype=tf.float32)
bia = tf.Variable(initial_value=[[4]], dtype=tf.float32)
y = tf.matmul(tf_X, weight) + bia

loss = tf.losses.mean_squared_error(tf_y, y)
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

X_data = np.linspace(-1, 1, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.1, X_data.shape)
y_data = X_data * real_params[0] + real_params[1] + noise

sess = tf.Session()
sess.run(tf.global_variables_initializer())

weights = []
biases = []
losses = []
for step in range(400):
    w, b, cost, _ = sess.run([weight, bia, loss, train_op],
                             feed_dict={tf_X: X_data, tf_y: y_data})
    weights.append(w)
    biases.append(b)
    losses.append(cost)
result = sess.run(y, feed_dict={tf_X: X_data, tf_y: y_data})

plt.figure(1)
plt.scatter(X_data, y_data, color='r', alpha=0.5)
plt.plot(X_data, result, lw=3)

fig = plt.figure(2)
ax_3d = Axes3D(fig)
w_3d, b_3d = np.meshgrid(np.linspace(-2, 7, 30), np.linspace(-2, 7, 30))
loss_3d = np.array(
    [np.mean(np.square((X_data * w_ + b_) - y_data))
     for w_, b_ in zip(w_3d.ravel(), b_3d.ravel())]).reshape(w_3d.shape)
ax_3d.plot_surface(w_3d, b_3d, loss_3d, cmap=plt.get_cmap('rainbow'))
weights = np.array(weights).ravel()
biases = np.array(biases).ravel()

# 描绘初始点
ax_3d.scatter(weights[0], biases[0], losses[0], s=30, color='r')
ax_3d.set_xlabel('w')
ax_3d.set_ylabel('b')
ax_3d.plot(weights, biases, losses, lw=3, c='r')

plt.show()
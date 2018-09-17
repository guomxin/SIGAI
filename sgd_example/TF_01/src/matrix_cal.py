import tensorflow as tf

diagonal = [1,1,1,1]

with tf.Session() as sess:
    print(sess.run(tf.diag(diagonal)))
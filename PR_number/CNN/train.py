import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import os

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
print(mnist.train.num_examples)
model_path = os.path.join("model", "model")
# 输入
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
# 真实值
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
# dropout
rate = tf.placeholder(dtype=tf.float32)

In = tf.reshape(x, [-1, 28, 28, 1])

net = slim.conv2d(In, 16, [13, 13], padding="VALID", scope='conv1')
net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
net = slim.conv2d(net, 32, [3, 3], padding="VALID", scope='conv2')
net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
net = tf.reshape(net, [-1, 3 * 3 * 32])
net = slim.fully_connected(net, 512)
net_drop = tf.nn.dropout(net, 1 - rate)
net = slim.fully_connected(net_drop, 10)
out = tf.nn.softmax(net)

loss = tf.reduce_mean(
    - tf.reduce_sum(y * tf.log(out), reduction_indices=[1])  # 先计算求和每行的交叉熵，然后去每个样本的交叉熵均值
)
global_step = tf.train.get_or_create_global_step()
lr = tf.train.exponential_decay(0.0002, global_step=global_step, decay_rate=0.9, decay_steps=550)
train_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=global_step, learning_rate=lr, optimizer="Adam")

with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=None)
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(30):
        for i in range(550):
            batchX, batchY = mnist.train.next_batch(100)
            sess.run(train_op, feed_dict={x: batchX, y: batchY, rate: 0.3})

        print(sess.run([loss, lr], feed_dict={x: batchX, y: batchY, rate: 0.3}))
        if (epoch + 1) % 10 == 0:
            saver.save(sess=sess, save_path=model_path, global_step=(global_step + 1))
            print("\nModel checkpoint saved...\n")
    print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels, rate: 0}))

import tensorflow as tf
import tensorflow.contrib.slim as slim


class CNN(object):
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.rate = tf.placeholder(dtype=tf.float32)
        self.modelinit()

    def modelinit(self):
        In = tf.reshape(self.x, [-1, 28, 28, 1])
        net = slim.conv2d(In, 16, [13, 13], padding="VALID", scope='conv1')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
        net = slim.conv2d(net, 32, [3, 3], padding="VALID", scope='conv2')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
        net = tf.reshape(net, [-1, 3 * 3 * 32])
        net = slim.fully_connected(net, 512)
        net_drop = tf.nn.dropout(net, 1 - self.rate)
        net = slim.fully_connected(net_drop, 10)

        self.out = tf.nn.softmax(net)
        self.loss = tf.reduce_mean(
            - tf.reduce_sum(self.y * tf.log(self.out), reduction_indices=[1])  # 先计算求和每行的交叉熵，然后去每个样本的交叉熵均值
        )
        self.global_step = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(0.0001, global_step=self.global_step, decay_rate=0.95, decay_steps=550)
        self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss, global_step=self.global_step,
                                                        learning_rate=self.lr,
                                                        optimizer="Adam")

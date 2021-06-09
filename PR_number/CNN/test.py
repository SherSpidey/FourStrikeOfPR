import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import pylab
from CNN.cnn import CNN

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
model_path = os.path.join("model", "Model")

x, y = mnist.test.next_batch(1)
print(y)
x = np.reshape(x, [28, 28])
print(x)
pylab.imshow(x)
pylab.show()
# model=CNN()
'''
with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=None)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess,model_path)

    bx,by=mnist.test.next_batch(1)
    oy=sess.run(tf.argmax(model.out,1),feed_dict={model.x: bx, model.rate: 0})
    print(oy)

    im = bx.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()'''

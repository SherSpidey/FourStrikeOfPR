# from CNN.cnn import CNN
# import Bayes.Bayes as bys
# import Fisher.Fisher as fsh
# import Perception.Pt as pt
# import tensorflow as tf
# import numpy as np
# import os
# #import cv2
#
# def load_data(num,num_dir='./Test'):
#     numdir = os.path.join(num_dir, "num{}.txt".format(num))
#     return np.loadtxt(numdir)
#
# def test(cfy='fisher'):
#     result=[]
#     if cfy=='cnn':
#         model = CNN()
#         with tf.Session() as sess:
#             saver = tf.train.Saver(max_to_keep=None)
#             init = tf.global_variables_initializer()
#             sess.run(init)
#             saver.restore(sess, os.path.join("CNN", "model", "Model"))
#             for i in range(10):
#                 pre=[0,0,0,0,0,0,0,0,0,0]
#                 num_data=load_data(i)
#                 for img in num_data:
#                     img = img.reshape(1, 784)
#                     out=sess.run(model.out, feed_dict={model.x: img, model.rate: 0})
#                     pre[np.argmax(out)]+=1
#                 result.append(pre)
#     if cfy=='bayes':
#         for i in range(10):
#             pre = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#             num_data = load_data(i)
#             for img in num_data:
#                 #img=cv2.resize(img,(50,70))由于整个项目完成时间的先后顺序，Bayes方法采用的特征远高于线性分类器，所以与CNN不在此处检验
#                 out = bys.test_no_G(img,os.path.join('Bayes','model.txt'),os.path.join('Bayes','num.txt'))
#                 pre[out] += 1
#             result.append(pre)
#     if cfy=='fisher':
#         for i in range(10):
#             pre = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#             num_data = load_data(i)
#             for img in num_data:
#                 out = fsh.test(img,'Fisher')
#                 pre[out] += 1
#             result.append(pre)
#     if cfy=='pt':
#         for i in range(10):
#             pre = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#             num_data = load_data(i)
#             for img in num_data:
#                 out = pt.test(img,'Perception')
#                 pre[out] += 1
#             result.append(pre)
#     result=np.array(result).reshape(10,-1)
#     return result
# print(test(cfy="pt"))
#
from PyQt5.QtCore import QT_VERSION_STR
from PyQt5.Qt import PYQT_VERSION_STR

print(QT_VERSION_STR)
print(PYQT_VERSION_STR)

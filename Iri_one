from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import tensorflow as tf 

import pandas as pd 

import numpy as np

from numpy import *

def one_hot_vector(true_label):

  #put every y into one-hot vector

  num=len(true_label)

  test=zeros((num,3))

  for i in range(num):#from 0 to num-1

    if true_label[i]==0:

      test[i,:]=[1,0,0]

    elif true_label[i]==1:

      test[i,:]=[0,1,0]

    else:

      test[i,:]=[0,0,1]

  return test

#load data





####training data

IRIS_TRAINING = '/home/liu/test/iris_training.csv'

marks = pd.read_csv(IRIS_TRAINING)

training=marks.ix[:,:4]

#training=reshape(training,(1,4))

y_label=marks.ix[:,4]

y_label=one_hot_vector(y_label)


####test data

IRIS_TEST = '/home/liu/test/iris_test.csv'

read_test = pd.read_csv(IRIS_TEST)

test=read_test.ix[:,:4]

#test=reshape(test,(1,4))

y_test=read_test.ix[:,4]

y_test=one_hot_vector(y_test)

#y_test=reshape(y_test,(1,3))



#####training

x = tf.placeholder(tf.float32, [None,4])

W = tf.Variable(tf.zeros([4, 3]))

b = tf.Variable(tf.zeros([3]))

y_ = tf.placeholder(tf.float32, [None,3])

y = tf.nn.softmax(tf.matmul(x, W) + b)



init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)



#training

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))



train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
for j in range(10):

  for i in range(120):

  #batch_xs, batch_ys = mnist.train.next_batch(2)

    sess.run(train_step, feed_dict={x: np.reshape(training.ix[i],(1,4)), y_: np.reshape(y_label[i],(1,3))})







#############test

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#print(sess.run(accuracy, feed_dict={x: test_.ix[1], y_: y_test[1]}))

print(sess.run(accuracy, feed_dict={x: reshape(test,(-1,4)), y_: reshape(y_test,(-1,3))}))










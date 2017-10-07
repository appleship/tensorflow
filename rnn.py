
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np 

batch_size=32
num_hidden=128
items=400

mnist = input_data.read_data_sets('/home/dingmengting/liu/python/mnist', one_hot=True)

x = tf.placeholder(tf.float32, [None, 28,28])
y = tf.placeholder(tf.float32, [None, 10])
# W = tf.Variable(tf.random_normal([28, 10]))
# b = tf.Variable(tf.random_normal([10]))

lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
#init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=None,dtype=tf.float32, time_major=False)
#results = tf.matmul(final_state[1], W) + b
results = tf.layers.dense(outputs[:, -1, :], 10) 

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=results))
train_op = tf.train.AdamOptimizer(0.001).minimize(cost)

correct_pred = tf.equal(tf.argmax(results,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(items):
    	batch_x, batch_y = mnist.train.next_batch(batch_size)
    	batch_x = batch_x.reshape((batch_size, 28, 28))
    	# out,final=sess.run([outputs,final_state], feed_dict={X: batch_x})
    	# print(out.shape)
    	# print(out)
    	# print(final.shape)
    	# print(final)
    	sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
    	if i%20==0:
    		loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y})
    		print('now is the %d term, the loss is %f, the acc is %f '%(i,loss,acc))
    print("training is over")
    print(sess.run(accuracy, feed_dict={x: mnist.test.images[:100].reshape(-1,28,28),y: mnist.test.labels[:100]}))








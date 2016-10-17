#Weight Initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)#随机分布的数列，标准方差 0.1
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)#bias全是常量0.1
  return tf.Variable(initial)

#Convolution and Pooling
 def conv2d(x, W):#其中strides=[1, 1, 1, 1]，第一个和第四个规定为1，x方向上跨度为1 y方向上跨度也是1
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#First Convolutional Layer

#patch 5*5  1是图片的厚度，32的高度，输出以后。
W_conv1 = weight_variable([5, 5, 1, 32])

b_conv1 = bias_variable([32])

#-1所有图片的维度先不管，后面再加上，最后一个1，是由于是黑白图像，如果是彩色图片的话，就要用3了。
x_image = tf.reshape(x, [-1,28,28,1])#28*28

#其中relu是非线性化的处理，也就是激励函数。因为padding=same，所以长宽不变。
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#此时图片变为 28*28*32

#由于stride=2，所以长宽减半，
h_pool1 = max_pool_2x2(h_conv1)#14*14*32

#Second Convolutional Layer

#上次处理完高度变成32，经处理变成64的高度，64这个可以随便写。
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)# 14*14*64
h_pool2 = max_pool_2x2(h_conv2)#7*7*64

#Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])#变成高度1024
b_fc1 = bias_variable([1024])

#由之前的【7 7 64】把它变平，变成7*7*64
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout，防止过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Train and Evaluate the Model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#用的优化函数是AdamOptimizer，因为数据量太庞大
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
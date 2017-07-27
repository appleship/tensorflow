import os

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
#from datagenerator import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICE"] = '1'

def read_and_decode(filename):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,features={
      'img_raw': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64) })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [227, 227, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label


batch_size = 32
filename="/home/liuchenlu/python/du/test.tfrecords"
filename_queue = tf.train.string_input_producer([filename])
img,label = read_and_decode(filename_queue)
#img_batch, label_batch = tf.train.shuffle_batch([img, label],batch_size=batch_size, capacity=100, min_after_dequeue=50,allow_smaller_final_batch=True)   
img_batch,label_batch = tf.train.batch([img, label], enqueue_many=True, batch_size=32, capacity=100, allow_smaller_final_batch=True)

x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
model = AlexNet(x)
score = model.fc6


init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
model.load_initial_weights(sess)
# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    #while not coord.should_stop():
    for j in range(10):
        i,l=sess.run([img_batch,label_batch])
        fea=sess.run(score, feed_dict={x: i})
        print(np.shape(fea))
        print(l)
    

except Exception, e:
    coord.request_stop(e)
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()


import os
import tensorflow as tf 
from PIL import Image

dir1='/home/liuchenlu/python/du/trainclass/0'
writer = tf.python_io.TFRecordWriter("/home/liuchenlu/python/du/test.tfrecords")


list1 = os.listdir(dir1)
for i in range(0,len(list1)):
    img_path=dir2+'/'+str(list1[i])
    img = Image.open(img_path)
    img = img.resize((227, 227))
	img_raw = img.tobytes()             
	example = tf.train.Example(features=tf.train.Features(feature={
    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
}))
	writer.write(example.SerializeToString())

writer.close()

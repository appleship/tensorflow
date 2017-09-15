import  pandas as pd 
import numpy as np 
import tensorflow as tf 

class DataSet(object):
	def __init__(self,images,labels):
		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def images(self):
    		return self._images

	@property
  	def labels(self):
    		return self._labels

	def next_batch(self,batch_size):
		start=self._index_in_epoch
		num=self._images.shape[0]
		#print('our',start,batch_size)
		if start==0:
			#num=self._images.shape[0]
			perm=np.arange(num)
			np.random.shuffle(perm)
			self._images=self.images[perm]
			self._labels=self.labels[perm]
			#print('first random')
		if start+batch_size>num:
			img_rest=self._images[start:,:]
			label_rest=self._labels[start:]
			new=batch_size-num+start
			perm=np.arange(num)
			np.random.shuffle(perm)
			self._images=self.images[perm]
			self._labels=self.labels[perm]
			#print('sec random')
			start=0
			img_new=self._images[start:new,:]
			label_new=self._labels[start:new]
			img_batch=np.concatenate((img_rest,img_new))
			label_batch=np.concatenate((label_rest,label_new))
			self._index_in_epoch=new
		else:
			#print(start)
			#print(perm[start:start+batch_size,:])
			#print(self._labels.shape)
			img_batch=self._images[start:start+batch_size,:]
			label_batch=self._labels[start:start+batch_size]
			self._index_in_epoch=start+batch_size
		#print(img_batch.shape)
		return img_batch,label_batch

def one_hot(label):
	classes=np.unique(label).shape[0]
	num=label.shape[0]
	new=np.zeros([num*classes])
	print new.shape
	index=np.arange(num)*classes+label.ravel()
	#print(index)
	new[index.astype('int64')]=1
	new=np.reshape(new,(num,classes))
	return new



def data():
	csv=pd.read_csv('/home/dingmengting/liu/python/iris.csv',header=None)
	csv=np.array(csv)
	perm=np.arange(50)
	np.random.shuffle(perm)
	tr_index=np.concatenate((perm[:40],perm[:40]+50,perm[:40]+100))
	te_index=np.concatenate((perm[40:],perm[40:]+50,perm[40:]+100))
	tr=csv[tr_index,:4]
	tr_label=one_hot(csv[tr_index,4])
	train=DataSet(tr,tr_label)
	te=csv[te_index,:4]
	te_label=one_hot(csv[te_index,4]) 
	test=DataSet(te,te_label)
	return train,test



def model(train,test):
	batch_size=32
	x = tf.placeholder(tf.float32, [None, 4])
	W = tf.Variable(tf.zeros([4, 3]))
	b = tf.Variable(tf.zeros([3]))
	y = tf.matmul(x, W) + b
	y_ = tf.placeholder(tf.float32, [None, 3])
	

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  	sess = tf.InteractiveSession()
  	tf.global_variables_initializer().run()
	for i in range(30):
		print('this is %s epoch' % i)

		batch_xs, batch_ys=train.next_batch(batch_size)
		sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

	correct_prediction=tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: test.images,
	                                  y_:test.labels}))



train,test=data()
model(train,test)




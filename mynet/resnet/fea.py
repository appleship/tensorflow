from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np

batch_size=32

#fea=np.load('/home/liuchenlu/python/test/fea_label.npz')
fea=np.load('/home/liuchenlu/python/test/test_fea.npz')
x=fea['fea']

# img_path = '/home/liuchenlu/python/du/trainclass/0/1003645637,3600678665.jpg'
# img = Image.open(img_path)
# img = img.resize((227, 227))
# img_ = np.array(img)
# x=img_* (1. / 255) - 0.5
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# print(np.shape(x))

model = ResNet50(include_top=False,weights='imagenet', input_shape=[227,227,3],
             pooling='avg')
fea = model.predict(x, batch_size=batch_size)
fea = fea.reshape((-1, 2048))
print(np.shape(fea))
np.savez('/home/liuchenlu/python/test/test_resnet.npz',pool=fea)

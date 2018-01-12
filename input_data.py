import skimage.io as io
import cv2
from skimage import data_dir
import tensorflow as tf

# file_path = "D:/final/try/*.png"
file_path = "D:\\final\\thumb"
# file_path = "1.TIF"


# collection=io.imread_collection(file_path)
# # # print(len(collection))
# i=0
# for i in range(100):
#     image=cv2.imread(file_path)
#     io.imshow(image)
str=file_path + '/*.png'
coll = io.ImageCollection(str)
print(len(coll))

h,w = coll[10000].shape[:2]
print(h,w)

x = tf.Variable(coll[20],name='x')
model = tf.initialize_all_variables()

with tf.Session() as session:
    x = tf.transpose(x,perm=[1,0])
    session.run(model)
    result = session.run(x)
cv2.namedWindow('image',0)
cv2.imshow('image',coll[10000])
cv2.waitKey(0)

image=cv2.resize(coll[10000],(400,400),interpolation=cv2.INTER_CUBIC)
cv2.imshow('image',image)
cv2.waitKey(0)
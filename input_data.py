import skimage.io as io
import cv2
from skimage import data_dir
import tensorflow as tf

# file_path = "D:/final/try/*.png"
file_path = "D:\\final\\try_100"
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

x = tf.Variable(coll[20],name='x')
model = tf.initialize_all_variables()

with tf.Session() as session:
    x = tf.transpose(x,perm=[1,0])
    session.run(model)
    result = session.run(x)

cv2.namedWindow('image',0)
cv2.imshow('image',coll[20])
cv2.waitKey(0)

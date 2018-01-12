import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


file_path = "D:\\final\\thumb\\00000000.png"
# file_path = "1.TIF"
image = cv2.imread(file_path,0)
image_after = image
h,w = image.shape[:2]
print(h,w)
sum = 0
for i in range(h):
    for j in range(w):
        sum+=image[i,j]
mean = sum/(h*w)
# for i in range(h):
#     for j in range(w):
#         image_after[i,j]-=mean
data1 = []
data2 = []
for i in range(w):
    print("1000",i,'--',image[300,i])
    data1.append(image[300,i])


for i in range(w):
    print("2000",i,'--',image[700,i])
    data2.append(image[700, i])

plt.plot(range(w),data1)
plt.savefig("data1.jpg")
plt.plot(range(w),data2)
plt.savefig("data2.jpg")

x = tf.Variable(image,name='x')
model = tf.initialize_all_variables()

with tf.Session() as session:
    x = tf.transpose(x,perm=[1,0])
    session.run(model)
    result = session.run(x)

cv2.namedWindow('image',0)
cv2.imshow('image',image)
cv2.waitKey(0)
# cv2.namedWindow('image_after',0)
# cv2.imshow('image_after',image_after)
# cv2.waitKey(0)
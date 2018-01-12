import skimage.io as io
import cv2

file_path = "D:/final/try/*.png"
file_path = "D:\\final\\thumb\\00000000.png"
# file_path = "1.TIF"

image = cv2.imread(file_path)


collection=io.imread_collection(file_path)
print(len(collection))
io.imshow(collection[20])
from skimage.io import imread
from scipy.ndimage import convolve, shift, label
from matplotlib.pyplot import imshow
from ipywidgets import interactive
from IPython.display import display
import matplotlib as mpl
import numpy as np

mpl.rc('image', cmap='gray')  # tell matplotlib to use gray shades for grayscale images
test_im = np.array(imread("what_is_a_river.jpg", as_gray=True)/255)  # This time the image is floating point 0.0 to 1.0!
height, width = test_im.shape
print("Test image shape: ", test_im.shape)
imshow(test_im)


import matplotlib.pyplot as plt
def edge_detection(int_list): #int_list is the weight matrix
  edgeimage_x = convolve(test_im,int_list, mode='constant', cval=0.0)
  edgeimage_y = convolve(test_im,int_list.T, mode='constant', cval=0.0)
  edgeimage = np.sqrt(edgeimage_x *edgeimage_x + edgeimage_y * edgeimage_y)

  for i in range(edgeimage.shape[0]):
    for j in range(edgeimage.shape[1]):
      if(edgeimage[i][j]> 0.0009): #threshhold is 100
        edgeimage[i][j] = 1
      else:
        edgeimage[i][j] = 0

  return edgeimage

elem_kernel = np.array([[-1,1]])
sobel_kernel= np.array([[-1,0,1],[-2, 0, 2],[-1,0,1]])
prewitt_kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(edge_detection(elem_kernel), cmap='gray')
plt.title("elem Edge Detection")
plt.axis('off')
# Sobel edge detection
plt.subplot(1, 3, 2)
plt.imshow(edge_detection(sobel_kernel), cmap='gray')
plt.title("Sobel Edge Detection")
plt.axis('off')

# Prewitt edge detection
plt.subplot(1, 3, 3)
plt.imshow(edge_detection(prewitt_kernel), cmap='gray')
plt.title("Prewitt Edge Detection")
plt.axis('off')

# Show the images
plt.show()

import numpy as np
from scipy.signal import fftconvolve
import scipy.misc
from scipy import ndimage
#import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfile

#load psf and image
object_path = askopenfile()
psf_path = askopenfile()
image_zemax_path = askopenfile()
object = np.loadtxt(object_path, skiprows=17, delimiter='\t')
image_output = Image.fromarray(object)
psf = np.loadtxt(psf_path, skiprows=19, delimiter='\t')
image_zemax = np.loadtxt(image_zemax_path, skiprows=17, delimiter='\t')
#convolve image with psf
convolved = fftconvolve(object, psf, mode='same')



plt.figure(figsize=(10,10))
plt.subplot(1,4,1)
plt.imshow(object, cmap='gray')
plt.title('Image')
plt.subplot(1,4,2)
plt.imshow(psf, cmap='gray')
plt.title('PSF')
plt.subplot(1,4,3)
plt.imshow(convolved, cmap='gray')
plt.title('Convolved Image')
plt.subplot(1,4,4)
img = image_zemax
rotated_img = ndimage.rotate(img, 180)
plt.imshow(rotated_img, cmap='gray')
plt.title('Zemax Simulation')
plt.show()

import numpy as np
from scipy.signal import fftconvolve
import cv2
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfile

#load psf and image
psf_path = askopenfile()
image_path = askopenfile()
psf = np.loadtxt(psf_path, skiprows=30, delimiter='\t')
image = np.loadtxt(image_path, skiprows=17, delimiter='\t')

#convolve image with psf
convolved = fftconvolve(image, psf, mode='same')

#plot image and convolved image
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Image')
plt.subplot(1,2,2)
plt.imshow(convolved, cmap='gray')
plt.title('Convolved Image')
plt.show()

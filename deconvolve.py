import numpy as np
from scipy import fftpack
# import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfile

# load object, psf and image
object_path = askopenfile()
psf_path = askopenfile()
image_zemax_path = askopenfile()

object = np.loadtxt(object_path, skiprows=17, delimiter='\t')
psf = np.loadtxt(psf_path, skiprows=18, delimiter='\t')
image_zemax = np.loadtxt(image_zemax_path, skiprows=17, delimiter='\t')

# data processing
object_normalized = object - np.min(object)
object_normalized = ((object_normalized / np.max(object_normalized)) * 255).astype(np.uint8)
#psf_normalized = psf - np.min(psf)
#psf_normalized = (psf_normalized / np.max(psf_normalized) * 100).astype(np.uint8)
image_zemax_normalized = image_zemax - np.min(image_zemax)
image_zemax_normalized = ((image_zemax_normalized / np.max(image_zemax_normalized)) * 255).astype(np.uint8)
print(object_normalized.shape)
print(np.min(object_normalized))
print(np.max(object_normalized))
print(psf.shape)
print(np.min(psf))
print(np.max(psf))
print(image_zemax_normalized.shape)
print(np.min(image_zemax_normalized))
print(np.max(image_zemax_normalized))


def convolve(object, psf):
    object_fft = fftpack.fftshift(fftpack.fftn(object))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(object_fft * psf_fft)))


def deconvolve(response, original):
    response_fft = fftpack.fftshift(fftpack.fftn(response))
    original_fft = fftpack.fftshift(fftpack.fftn(original))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(response_fft / original_fft)))

# convolution between object and PSF
conv = convolve(object_normalized, psf)
# deconvolution between convolution and object
deconv = deconvolve(conv, object_normalized)
# deconvolution between image and object
deconv2 = deconvolve(image_zemax_normalized, object_normalized)

# deconv2_normalized = deconv2 - np.min(deconv2)
# deconv2_normalized = (deconv2_normalized/np.max(deconv2_normalized)) * np.max(conv)

fig = plt.figure()
plt.subplot(3, 2, 1)
plt.imshow(object_normalized)
plt.title('Object')
plt.subplot(3, 2, 2)
plt.imshow(psf)
plt.title('PSF')
plt.subplot(3, 2, 3)
plt.imshow(np.real(conv))
plt.title('C = Convolution(Object*PSF)')
plt.subplot(3, 2, 4)
plt.imshow(np.real(deconv))
plt.title('Deconvolution(C/Object)')
plt.subplot(3, 2, 5)
plt.imshow(image_zemax_normalized)
plt.title('IZ = Image Zemax')
plt.subplot(3, 2, 6)
plt.imshow(np.real(deconv2))
plt.title('Deconvolution(IZ/Object)')
plt.show()

# f, axes = plt.subplots(3,2)
# axes[0,0].imshow(object_normalized)
# axes[0,1].imshow(psf_normalized)
# axes[1,0].imshow(np.real(conv))
# axes[1,1].imshow(np.real(deconv))
# axes[2,0].imshow(image_zemax_normalized)
# axes[2,1].imshow(np.real(deconv2))
# plt.show()

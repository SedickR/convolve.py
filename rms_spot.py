import cv2
import numpy as np
import os


def center_of_mass(path):
    '''Arguement: path to image
    Returns : coordinates of center of mass, image in grayscale'''
    # read image through command line
    img = cv2.imread(f'{path}')

    # convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert the grayscale image to binary image
    ret,thresh = cv2.threshold(gray_image,10,255,0)

    # create a meshgrid for coordinate calculation
    x_m, y_m = np.meshgrid(
            np.arange(0, thresh.shape[1]),
            np.arange(0, thresh.shape[0]),
            sparse=False,
            indexing='xy'
        )

    x_c = int(np.sum(x_m * thresh) / np.sum(thresh))
    y_c = int(np.sum(y_m * thresh) / np.sum(thresh))

    return (x_c, y_c), thresh


def rms_spot(center, img_gray):
    '''Arguement: center of mass, grayscale image
    Returns : rms spot size'''
    # Find all non-zero pixels in the image
    non_zero = cv2.findNonZero(img_gray)

    # Distance of all non-zero pixels from the center of mass
    dist = np.sqrt((non_zero[:,:,0] - center[0])**2 + (non_zero[:,:,1] - center[1])**2)
    
    # Return the rms spot size
    return np.sqrt(np.mean(dist**2))




for filename in os.listdir('images'):
    cv2.imshow('image', center_of_mass(f'images\\{filename}')[1])
    cv2.waitKey(0)


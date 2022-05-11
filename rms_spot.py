import cv2 as cv
import numpy as np
import os
import yaml


class image_analysis:
    def __init__(self, path):
        if path.endswith('.bmp'):
            self.directory = os.path.dirname(path)
            self.image = cv.imread(path)
        else:
            self.directory = path
        self.background = None

    def background_subtraction(self, image, background):
        '''Arguement: image, background
        Return: subtracted image'''
        return cv.subtract(image, background)

    def center_of_mass(self, im_path, channel="r", background_path=None):
        '''Arguement: path to image, color channel, path to background image
        Returns : coordinates of center of mass, image in grayscale, [pixel width, unit]'''
        # get display data
        with open("display_data.yaml", 'r') as f:
            display = yaml.safe_load(f)

        # read image through command line
        img = cv.imread(f'{im_path}')

        cv.imshow("image", img)
        cv.waitKey(0)

        #remove background if background path is given
        if background_path:
            background = cv.imread(f'{background_path}')
            img = self.background_subtraction(img, background)
        
        cv.imshow("image", img)
        cv.waitKey(0)

        # split image into channels
        b, g, r = cv.split(img)
        if channel is "b":
            _,thresh = cv.threshold(b,15,255,0)
            color = 'blue'
        elif channel is "g":
            _,thresh = cv.threshold(g,15,255,0)
            color = 'green'
        else:
            _,thresh = cv.threshold(r,15,255,0)
            color = 'red'

        cv.imshow("image", thresh)
        cv.waitKey(0)

        # define center of image
        center = (int(thresh.shape[1]//2), int(thresh.shape[0]//2))

        # mask unwanted pixels based on display measurements
        cv.circle(thresh, (center[0]-display[color], center[1]+display[color]), display['noise_px_diameter'], (0,0,0), -1)
        cv.circle(thresh, (center[0]+display[color], center[1]+display[color]), display['noise_px_diameter'], (0,0,0), -1)
        cv.circle(thresh, (center[0]-display[color], center[1]-display[color]), display['noise_px_diameter'], (0,0,0), -1)
        cv.circle(thresh, (center[0]+display[color], center[1]-display[color]), display['noise_px_diameter'], (0,0,0), -1)

        cv.imshow("image", thresh)
        cv.waitKey(0)

        # create a meshgrid for coordinate calculation
        x_m, y_m = np.meshgrid(
                np.arange(0, thresh.shape[1]),
                np.arange(0, thresh.shape[0]),
                sparse=False,
                indexing='xy'
            )

        # compute center of mass
        x_c = int(np.sum(x_m * thresh) / np.sum(thresh))
        y_c = int(np.sum(y_m * thresh) / np.sum(thresh))


        return (x_c, y_c), thresh, display['pixel_diameter']


    def rms_spot(self, center, img_gray, units):
        '''Arguement: center of mass, grayscale image, [pixel width, unit]
        Returns : rms spot size, units'''
        # Find all non-zero pixels in the image
        non_zero = cv.findNonZero(img_gray)

        # Distance of all non-zero pixels from the center of mass
        dist = np.sqrt((non_zero[:,:,0] - center[0])**2 + (non_zero[:,:,1] - center[1])**2)

        # Return the rms spot size
        return np.sqrt(np.mean(dist**2))*units[0], units[1]


i = image_analysis('red_spot_size_xrot\\background_spm.bmp')

print(i.rms_spot(*i.center_of_mass('red_spot_size_xrot\\2h_x-1,5081d_y0d.bmp', 'r', 'red_spot_size_xrot\\background_spm.bmp')))
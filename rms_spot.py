import cv2 as cv
import numpy as np
import os, yaml, time, csv


class image_analysis:
    def __init__(self, path):
        self.channel = None
        if path.endswith('.bmp') or path.endswith('.png'):
            self.directory = os.path.dirname(path)
            self.image_path = path
            print('Single image mode')
        else:
            self.image_path = None
            self.directory = path

        # retrieve background image, and display data if they exist
        self.display_path = None
        self.background_path = None
        self.background = None
        self.display_path = "default_display.yaml"
        try:
            for file in os.listdir(self.directory):
                if 'background' in file:
                    self.background_path = os.path.join(self.directory, file)
                    print('Background image found')
                if 'display' in file:
                    self.display_path = os.path.join(self.directory, file)
                    print('display data found')
        except FileNotFoundError:
            print('No background image or display data found')

        

    def background_subtraction(self, image, background):
        '''Arguement: image, background
        Return: subtracted image'''
        return cv.subtract(image, background)

    def center_of_mass(self, im_path, display_data, channel="r", background_path=None, show=False):
        '''Arguement: path to image, color channel, path to background image
        Returns : coordinates of center of mass, image in grayscale, [pixel width, unit]'''
        # get display data
        with open(display_data, 'r') as f:
            display = yaml.safe_load(f)

        # read image through command line
        img = cv.imread(f'{im_path}')

        if show:
            cv.imshow("image", img)
            cv.waitKey(0)

        #remove background if background path is given
        if background_path:
            background = cv.imread(f'{background_path}')
            img = self.background_subtraction(img, background)
        
        if show:
            cv.imshow("image", img)
            cv.waitKey(0)

        # split image into channels
        b, g, r = cv.split(img)
        if channel == "b":
            _,thresh = cv.threshold(b,15,255,0)
            color = 'blue'
        elif channel == "g":
            _,thresh = cv.threshold(g,15,255,0)
            color = 'green'
        else:
            _,thresh = cv.threshold(r,15,255,0)
            color = 'red'
        
        if show:
            cv.imshow("image", thresh)
            cv.waitKey(0)

        # define center of image
        center = (int(thresh.shape[1]//2), int(thresh.shape[0]//2))

        # mask unwanted pixels based on display measurements
        cv.circle(thresh, (center[0]-display[color][1], center[1]+display[color][0]), display['noise_px_diameter'], (0,0,0), -1)
        cv.circle(thresh, (center[0]+display[color][0], center[1]+display[color][1]), display['noise_px_diameter'], (0,0,0), -1)
        cv.circle(thresh, (center[0]-display[color][0], center[1]-display[color][1]), display['noise_px_diameter'], (0,0,0), -1)
        cv.circle(thresh, (center[0]+display[color][1], center[1]-display[color][0]), display['noise_px_diameter'], (0,0,0), -1)
        if show:
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
        return np.sqrt(np.mean(dist**2))*units[0]

    def compute_rms(self, channel="r", show=False):
        '''Returns: [(file_name, rms spot size)]'''
        result = []
        self.channel = channel
        if self.image_path:
            result.append([file, self.rms_spot(*self.center_of_mass(self.image_path, self.display_path, channel, self.background_path, show))])
        else:
            for file in os.listdir(self.directory):
                if file.endswith('.bmp') and 'background' not in file:
                    computed = self.rms_spot(*self.center_of_mass(os.path.join(self.directory, file), self.display_path, channel, self.background_path, show))
                    if show:
                        print(f'{file}: {computed[0]} {computed[1]}')
                    result.append([file, computed])
        if result is None:
            raise Exception("No images found in specified directory")
        result = np.array(result)
        result = result[np.argsort(result[:,0])]
        return result
    
    def save_to_csv(self, result, filename):
        '''Arguement: result, filename
        Returns: None'''
        column_index = {'r': 3, 'g': 2, 'b': 1}
        with open(filename, 'w') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['Spot Size', 'B Spot Size', 'G Spot Size', 'R Spot Size'])
            for row in result:
                row_template = [0, 0, 0, 0]
                row_template[column_index[self.channel]] = row[1]
                writer.writerow(row_template)


i = image_analysis('images')
print(i.compute_rms(channel = 'r'))
i.save_to_csv(i.compute_rms(channel = 'r'), 'test1.csv')

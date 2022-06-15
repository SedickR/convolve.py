import cv2
import numpy as np
import os, yaml, time, csv
import pathlib
from tkinter.filedialog import askopenfile
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt


class image_analysis:
    """
    The class implements a series of method to analyze point size data of
    sub-pixel imaged through a MLA. It allows to quantify the spot size of
    a display and lens array using a series of digital images.
    """

    def __init__(self, path: pathlib.Path = None, img: np.array = None):
        """Arguement: path to image or directory containing images, np.array or cv2 representing image"""

        # Class variable initialization
        self.img = None  # image
        self.channel = "r"  # color channel
        self.image_path = None  # path to image
        self.background_path = None  # path to background image
        self.background = None  # background image
        self.directory = None  # path to directory containing images
        self.display_path = pathlib.Path("default_display.yaml")  # path to display data

        try:
            path = pathlib.Path(path)
            if path.suffix == ".bmp" or path.suffix == ".png":
                self.directory = path.parent
                self.image_path = path
                print("Single image mode")
            else:
                self.directory = path
            # retrieve background image, and display data if they exist
            try:
                for file in self.directory.iterdir():
                    if "background" in str(file):
                        self.background_path = file
                        print("Background image found")
                    if "display" in str(file):
                        self.display_path = file
                        print("display data found")
            except FileNotFoundError:
                print("No background image or display data found, using default")
        except TypeError:
            print("No path given, using direct image mode")
            self.img = img

    def center_of_mass_cca(
        self,
        im_path,
        display_data,
        channel="r",
        background_path=None,
        show=False,
        img=None,
    ):
        """Arguement: path to image, yaml display data file path, color channel, path to background image, verbosity, image (optional)
        Returns : coordinates of center of mass, image in grayscale, [pixel width, unit], img"""
        # get display data
        with open(display_data, "r") as f:
            display = yaml.safe_load(f)

        # read image
        if im_path:
            img = cv2.imdecode(
                np.fromfile(im_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )

        if show:
            cv2.imshow("Original", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # remove background if background path is given
        if background_path:
            background = cv2.imread(rf"{background_path}")
            img = cv2.subtract(img, background)

        if show and background_path:
            cv2.imshow("Substracted", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        value = 5  # Value of the threshold

        # split image into channels
        b, g, r = cv2.split(img)
        if channel == "b":
            _, thresh = cv2.threshold(b, value, 255, 0)
            color = "blue"
            img = b
        elif channel == "g":
            _, thresh = cv2.threshold(g, value, 255, 0)
            color = "green"
            img = g
        else:
            _, thresh = cv2.threshold(r, value, 255, 0)
            color = "red"
            img = r

        if show:
            cv2.imshow("Thresholed", thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Components analysis of thresholded image
        analysis = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
        labels = analysis[1]  # labels of connected components
        stats = analysis[2][1:]  # stats of connected components
        centroids = analysis[3][1:]  # centroids of connected components


        # Find the largest component's index
        largest_component = np.argmax(stats[:, cv2.CC_STAT_AREA])
        largest_component_centroid = centroids[largest_component]



        # Show label image of largest component
        label_image = np.zeros(thresh.shape, dtype=np.uint8)
        label_image[labels == largest_component + 1] = 255

        if show:
            cv2.imshow("Filtered", label_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return (
            (largest_component_centroid[0], largest_component_centroid[1]),
            label_image,
            display["pixel_diameter"],
            img,
        )

    def rms_spot(self, center, img_gray, units):
        """Arguement: center of mass, grayscale image, [pixel width, unit]
        Returns : rms spot size, units"""
        # Find all non-zero pixels in the image
        non_zero = cv2.findNonZero(img_gray)

        # Distance of all non-zero pixels from the center of mass
        dist = np.sqrt(
            (non_zero[:, :, 0] - center[0]) ** 2 + (non_zero[:, :, 1] - center[1]) ** 2
        )

        # Return the rms spot size
        return np.sqrt(np.mean(dist**2)) * units[0]
        # return np.sqrt(np.sum(img[non_zero[:, :, 1], non_zero[:, :, 0]]*dist**2)/np.sum(img[non_zero[:, :, 1], non_zero[:, :, 0]])) * units[0]

    def compute_rms(self, channel="r", show=False):
        """Returns: [(file_name, rms spot size)]

        Computes the rms spot size for all images in the directory or for a single image.
        If none of the above is given, and a self.image is given, the rms spot size is computed for that image."""
        result = []
        self.channel = channel  # Retrieve color channel

        # If a single image is given, compute the rms spot size for that image
        if self.image_path:
            computed = [
                self.image_path.name,
                self.rms_spot(
                    *self.center_of_mass_cca(
                        self.image_path,
                        self.display_path,
                        channel,
                        self.background_path,
                        show,
                    )
                ),
            ]
            if show:
                print(f"{computed[0]}: {computed[1]}")
            result.append(computed)

        # If a directory is given, compute the rms spot size for all images in the directory
        elif self.directory:
            for file in self.directory.iterdir():
                if file.suffix == ".bmp" and "background" not in str(file):
                    computed = self.rms_spot(
                        *self.center_of_mass_cca(
                            file,
                            self.display_path,
                            channel,
                            self.background_path,
                            show,
                        )
                    )
                    if show:
                        print(f"{file}: {computed}")
                    result.append([file, computed])

        # If no image is given, compute the rms spot size for the image in the class
        else:
            computed = self.rms_spot(
                *self.center_of_mass_cca(
                    self.image_path,
                    self.display_path,
                    channel,
                    self.background_path,
                    show,
                    self.img,
                )
            )
            if show:
                print(f"Result (um): {computed}")
            return computed
        if result == []:
            raise Exception("No images found in specified directory")

        return result[np.argsort(np.array(result)[:, 0])]

    def compute_lateral_color(self, red, blue, show=False):
        """Arguments: red image, blue image, verbosity

        Returns: dist"""
        center_red, _, _, _ = self.center_of_mass_cca(
            self.image_path, self.display_path, "r", self.background_path, show, red
        )  # Compute red image centroid

        center_blue, _, _, _ = self.center_of_mass_cca(
            self.image_path, self.display_path, "b", self.background_path, show, blue
        )  # Compute blue image centroid
        print(f"centers : {center_red, center_blue}")

        dist = np.sqrt(
            (center_red[0] - center_blue[0]) ** 2
            + (center_red[1] - center_blue[1]) ** 2
        )

        return dist

    def save_to_csv(self, result, filename, full=None):
        """Arguement: result, filename
        Returns: None"""
        column_index = {
            "r": 3,
            "g": 2,
            "b": 1,
        }  # Specific column index for each channel
        with open(filename, "w") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Spot Size", "B Spot Size", "G Spot Size", "R Spot Size"])
            for row in result:
                row_template = [0, 0, 0, 0]
                if full:
                    row_template[column_index[self.channel]] = row
                else:
                    row_template[column_index[self.channel]] = row[1]
                writer.writerow(row_template)

    def complete_ei(self, channel, show):
        """
        Argument: target_file

        Takes a directory of images in vertical, horizontal and diagonal directions and returns
        a formatted csv file with the rms spot size for each image.
        """
        # check for diagonal, horizontal and vertical directory paths
        diagonal_path = self.directory / "diagonal"
        horizontal_path = self.directory / "horizontal"
        vertical_path = self.directory / "vertical"
        if not (
            diagonal_path.exists()
            and horizontal_path.exists()
            and vertical_path.exists()
        ):
            raise Exception(
                "No diagonal, horizontal or vertical directory found in the specified directory"
            )

        # compute rms for each directory
        self.directory = diagonal_path
        diagonal_result = self.compute_rms(channel, show)[:, 1]
        self.directory = horizontal_path
        horizontal_result = self.compute_rms(channel, show)[:, 1]
        self.directory = vertical_path
        vertical_result = self.compute_rms(channel, show)[:, 1]

        # combine result in a single array
        size = int(
            min(len(diagonal_result), len(horizontal_result), len(vertical_result))
        )
        array = np.zeros((size, size))
        array[0] = horizontal_result[:size]
        array[:, 0] = vertical_result[:size]
        array[range(size), range(size)] = diagonal_result[:size]
        return array.flatten()

    def map_range(self, x, in_min, in_max, out_min, out_max):
            """Map a value from one range to another"""
            return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

    def convolve(self, show=False):
        #load psf and image
        psf_path = askopenfile()
        image_path = askopenfile()
        psf = np.loadtxt(psf_path, skiprows=21, delimiter='\t')
        image = np.loadtxt(image_path, skiprows=17, delimiter='\t')

        #convolve image with psf
        convolved = fftconvolve(image, psf, mode='same')
        convolved = self.map_range(convolved, np.amin(convolved), np.amax(convolved), 0, 255)
        #plot image, psf and convolved image
        if show:
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Image')
            plt.subplot(1, 3, 2)
            plt.imshow(psf, cmap='gray')
            plt.title('PSF')
            plt.subplot(1, 3, 3)
            plt.imshow(convolved, cmap='gray')
            plt.title('Convolved Image')
            plt.show()
        
        
        #map convolved to an rgb image
        rgb = np.zeros((image.shape[0], image.shape[1], 3))
        print(convolved)
        rgb[:,:,2] = convolved
        return rgb

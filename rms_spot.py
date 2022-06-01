import cv2 as cv
import numpy as np
import os, yaml, time, csv
import pathlib, typing


class image_analysis:
    """
    The class implements a series of method to analyze point size data of
    sub-pixel imaged through a MLA. It allows to quantify the spot size of
    a display and lens array using a series of digital images.
    """

    def __init__(self, path: pathlib.Path):
        """Arguement: path to image or directory containing images"""
        path = pathlib.Path(path)

        self.channel = None
        if path.suffix == ".bmp" or path.suffix == ".png":
            self.directory = path.parent
            print(type(self.directory))
            self.image_path = path
            print("Single image mode")
        else:
            self.image_path = None
            self.directory = path
        # retrieve background image, and display data if they exist
        self.display_path = None
        self.background_path = None
        self.background = None
        self.display_path = pathlib.Path("default_display.yaml")
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


    def center_of_mass_cca(
        self, im_path, display_data, channel="r", background_path=None, show=False
    ):
        """Arguement: path to image, yaml display data file path, color channel, path to background image
        Returns : coordinates of center of mass, image in grayscale, [pixel width, unit]"""
        # get display data
        with open(display_data, "r") as f:
            display = yaml.safe_load(f)

        # read image
        img = cv.imread(rf"{im_path}")

        if show:
            cv.imshow("image", img)
            cv.waitKey(0)

        # remove background if background path is given
        if background_path:
            background = cv.imread(rf"{background_path}")
            img = cv.subtract(img, background)

        if show:
            cv.imshow("image", img)
            cv.waitKey(0)

        value = 4

        # split image into channels
        b, g, r = cv.split(img)
        if channel == "b":
            _, thresh = cv.threshold(b, value, 255, 0)
            color = "blue"
        elif channel == "g":
            _, thresh = cv.threshold(g, value, 255, 0)
            color = "green"
        else:
            _, thresh = cv.threshold(r, value, 255, 0)
            color = "red"

        if show:
            cv.imshow("image", thresh)
            cv.waitKey(0)

        # Components analysis of thresholded image
        analysis = cv.connectedComponentsWithStats(thresh, 8, cv.CV_32S)
        labels = analysis[1]
        stats = analysis[2]
        centroids = analysis[3]

        # Remove first component (background)
        stats = stats[1:]
        centroids = centroids[1:]

        # Find the largest component's index
        largest_component = np.argmax(stats[:, cv.CC_STAT_AREA])
        largest_component_centroid = centroids[largest_component]

        # Show label image of largest component
        label_image = np.zeros(thresh.shape, dtype=np.uint8)
        label_image[labels == largest_component + 1] = 255

        if show:
            cv.imshow("image", label_image)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return (
            (int(largest_component_centroid[0]), int(largest_component_centroid[1])),
            label_image,
            display["pixel_diameter"]
        )

    def rms_spot(self, center, img_gray, units):
        """Arguement: center of mass, grayscale image, [pixel width, unit]
        Returns : rms spot size, units"""
        # Find all non-zero pixels in the image
        non_zero = cv.findNonZero(img_gray)

        # Distance of all non-zero pixels from the center of mass
        dist = np.sqrt(
            (non_zero[:, :, 0] - center[0]) ** 2 + (non_zero[:, :, 1] - center[1]) ** 2
        )

        # Return the rms spot size
        return np.sqrt(np.mean(dist**2)) * units[0]

    def compute_rms(self, channel="r", show=False):
        """Returns: [(file_name, rms spot size)]

        Computes the rms spot size for all images in the directory or for a single image."""
        result = []
        self.channel = channel
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
                    ))]
            if show:
                print(f'{computed[0]}: {computed[1]}')
            result.append(computed)
        else:
            for file in self.directory.iterdir():
                if file.suffix == ".bmp" and "background" not in str(file):
                    computed = self.rms_spot(
                        *self.center_of_mass_cca(
                            file,
                            self.display_path,
                            channel,
                            self.background_path,
                            show,
                        ))
                    if show:
                        print(f"{file}: {computed}")
                    result.append([file, computed])
        if result == []:
            raise Exception("No images found in specified directory")
        result = np.array(result)
        result = result[np.argsort(result[:, 0])]
        return result

    def save_to_csv(self, result, filename, full=None):
        """Arguement: result, filename
        Returns: None"""
        column_index = {"r": 3, "g": 2, "b": 1}
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

        # compute rmss for each directory
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

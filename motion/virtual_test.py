from cam import get_camera, setup_camera
from vimba import *
import cv2 as cv
import yaml
import numpy as np



def get_frame():
    with Vimba.get_instance():
        with get_camera(None) as cam:
            #setup camera
            setup_camera(cam)

            #Save a single frame to opencv format
            frame = cam.get_frame()
            

            #Show the frame
            cv.imshow('frame', frame.as_opencv_image())
            cv.waitKey(0)
        
    return frame.as_opencv_image()

def center_of_mass_cca(
        im_path, display_data, channel="r", background_path=None, show=False
    ):
    """Arguement: path to image, yaml display data file path, color channel, path to background image
    Returns : coordinates of center of mass, image in grayscale, [pixel width, unit]"""
    # get display data
    with open(display_data, "r") as f:
        display = yaml.safe_load(f)
    # read image
    img = im_path
    if show:
        cv.imshow("image", img)
        cv.waitKey(0)
    value = 15
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
    return (
        (int(largest_component_centroid[0]), int(largest_component_centroid[1])),
        label_image,
        display["pixel_diameter"],
    )


print(center_of_mass_cca(get_frame(), 'default_display.yaml', 'r', None, True))


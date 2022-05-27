from audioop import avg
from mailbox import linesep
from platform import platform
from tkinter import N, Y
from cam import get_camera, setup_camera
from vimba import *
import cv2 as cv
import yaml, sys, cam

sys.path.insert(0, r'C:\Users\emile\OneDrive\Université\Stages\3D écran\InImg\motion')

from motion_class import CustomMotion
import numpy as np
import poly_point_isect as bot
from zaber_motion import Units




def get_frame():
    with Vimba.get_instance():
        with get_camera(None) as cam:
            #setup camera
            setup_camera(cam)
            #Save a single frame to opencv format
            frame = cam.get_frame()
    return frame.as_opencv_image()

def get_intersection(img):
    '''Coordinates of lines intersection
    
    Code source https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv'''
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    kernel_size = 7
    blur_gray = cv.GaussianBlur(gray,(kernel_size, kernel_size),0)

    low_threshold = 50
    high_threshold = 150
    edges = cv.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    #print(lines)
    points = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    lines_edges = cv.addWeighted(img, 0.8, line_image, 1, 0)
    #print(lines_edges.shape)
    #cv.imwrite('line_parking.png', lines_edges)

    #print(points)
    intersections = bot.isect_segments(points)
    #print(intersections)

    for inter in intersections:
        a, b = inter
        for i in range(3):
            for j in range(3):
                lines_edges[int(b) + i, int(a) + j] = [0, 255, 0]

    return intersections, lines_edges

def get_center(img):

    #extract lumincance chanel
    _, _, img = cv.split(img)

    value = 50
    _, thresh = cv.threshold(img, value, 255, 0)
    #show(thresh)

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
    return (
        (int(largest_component_centroid[0]), int(largest_component_centroid[1])),
        label_image
    )

def show(img):
    scale_percent = 30 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    cv.imshow('frame', resized)
    cv.waitKey(0)
    cv.destroyAllWindows()


def distance(img):
    '''Aligns the camera to the intersection of target'''

    #get frame
    #img = get_frame()

    #get intersection
    intersections, lines_edges = get_center(img)

    #Compute average of the four intersections


    #define center of the frame
    center = (int(img.shape[1]/2), int(img.shape[0]/2))

    #Compute x and y disance of center to intersection
    x_dis = intersections[0] - center[0]
    y_dis = center[1] - intersections[1]

    print(f'The center is {center}', f'The intersection is {intersections}')

    #Add line frome center to intersection
    cv.line(lines_edges, intersections, center, (0, 0, 100), 5)

    print(f'The x y distances are {x_dis, y_dis}')

    #display image
    #show(lines_edges)
    

    return x_dis, y_dis

def get_scale(img, motion:CustomMotion):
    x_before,_ = distance(img)
    motion.x_axis.move_relative((np.sign(x_before))*-10000, Units.LENGTH_MICROMETRES)
    img = get_frame()
    x_after,_ = distance(img)
    scale = abs(10000 / (x_after - x_before))
    print(f'The scale at z position {motion.z_axis.get_position(Units.LENGTH_MILLIMETRES)} is {scale} micrometers per pixel')
    return scale, img


def bold_adjust():

    #Display a frame
    img = get_frame()
    #show(img)

    target = input('Is the target in the frame? (y/n)')
    if target == 'y':
        pass
    else:
        print('Please place the target in the frame then press enter')
        cam.main()
        cv.destroyAllWindows()
    
    img = get_frame()
    
    #Initialize motion platform
    motion = CustomMotion(False, 'com5', 'com7')
    
    z_position = np.round(motion.z_axis.get_position(Units.LENGTH_MILLIMETRES), 2)
    
    #get pixel scale
    with open(r'motion\scales.yaml', 'r') as file:
        scales = yaml.safe_load(file)
    
    if f'{z_position}' in scales:
        scale = scales[f'{z_position}']
        overwrite = False
    else:
        scale, img = get_scale(img, motion)
        scales[f'{z_position}'] = scale
        overwrite = True
    
    if overwrite:
        with open(r'motion\scales.yaml', 'w') as file:
            yaml.safe_dump(scales, file)
    
    
    #Move x and y axis to center the target
    x, y = distance(img)
    motion.x_axis.move_relative(-x*scale, Units.LENGTH_MICROMETRES, False)
    motion.y_axis.move_relative(y*scale, Units.LENGTH_MICROMETRES, False)

    motion.x_axis.wait_until_idle()
    motion.y_axis.wait_until_idle()

    img = get_frame()
    #crop image
    img = img[872:1073, 1196:1397, :]
    #show(img)
    #Move x and y axis to center the target
    x, y = distance(img)
    motion.x_axis.move_relative(-x*scale, Units.LENGTH_MICROMETRES, False)
    motion.y_axis.move_relative(y*scale, Units.LENGTH_MICROMETRES, False)
    motion.x_axis.wait_until_idle()
    motion.y_axis.wait_until_idle()
    show(get_frame())
    
    motion.disconnect()
    

bold_adjust()
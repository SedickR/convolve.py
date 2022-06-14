from cv2 import MOTION_TRANSLATION
from regex import P
from zaber_motion import Library
from zaber_motion.ascii import Connection
from zaber_motion import Units
import asyncio, time
from zaber_motion.ascii import SettingConstants
import numpy as np

Library.enable_device_db_store()

# Token nhqcQy6_KpchnMzV6Q-9v4HCRG8zAgtB


class CustomMotion():
    def __init__(self, home=False, *ports:str):
        '''Arguments: Serial ports
        
        Initialize the serial port commmunication with zabber motion plateforms
        and assigns each axis to the corresponding variable'''
        self.connection1 = Connection.open_serial_port(ports[0])
        self.connection2 = Connection.open_serial_port(ports[1])
        device_list = []
        device_list += self.connection1.detect_devices()
        device_list += self.connection2.detect_devices()
        #Expected devices
        self.x_axis = None
        self.y_axis = None
        self.z_axis = None
        self.pitch = None
        self.yaw = None
        
        #assign each device to its axis
        for device in device_list:
            if 'RST120' in device.__str__():
                if self.yaw:
                    self.pitch = device.get_axis(1)
                else:
                    self.yaw = device.get_axis(1)

            if '450' in device.__str__():
                self.z_axis = device.get_axis(1)

            if '300' in device.__str__():
                self.x_axis = device.get_axis(1)
            
            if 'VSR' in device.__str__():
                self.y_axis = device.get_axis(1)
        
        self.axis_list = [self.x_axis, self.y_axis, self.z_axis]
        self.rotary_list = [self.pitch, self.yaw]
        self.all_axes = [self.x_axis, self.y_axis, self.z_axis, self.pitch, self.yaw]
        self.axis_list = list(filter(None, self.axis_list))
        self.rotary_list = list(filter(None, self.rotary_list))
        self.all_axes = list(filter(None, self.all_axes))
        # check all position and ensure they are positive (smaller than 180), then home

        if home:
            for axis in self.axis_list:
                axis.home(False)

            for axis in self.rotary_list:
                self.home_rotation(axis)

            for axis in self.all_axes:
                axis.wait_until_idle()

    def move_z(self, position, wait=True):
        '''Arguments: position

        Checks for conflict with gimbal then move the z axis to the position.
        If a conflict is detected, the z axis is moved to the closest position'''
        yaw = abs(self.yaw.get_position(Units.ANGLE_RADIANS))
        if yaw > np.pi/2:
            yaw = yaw - (yaw-np.pi/2)
        print(f'yaw : {yaw}')

        a = 225*np.sin(yaw)
        b = 229.25834*np.sin(yaw + 0.195258)
        print(f'Distance from end a {a-79.5} and distance from end b is {b-79.5}')
        distance = 450-max(a,b)+78
        print(f'Max distance is {distance}')
        if a < 528-position and b < 528-position:
            self.z_axis.move_absolute(position, Units.LENGTH_MILLIMETRES, wait)
            print('The movement is allowed')
        else:
            print('Conflict between gimbal and z axis, making longest movement possible')
            self.z_axis.move_absolute(distance, Units.LENGTH_MILLIMETRES, wait)
            print('Moving to distance')

    def move_yaw(self, position):
        """Arguments: position in degrees
        
        Checks for conflict with z axis then move the yaw axis to the position.
        If a conflict is detected, the z axis is moved accordingly, then the yaw axis is moved to the position."""
        z = self.z_axis.get_position(Units.LENGTH_MILLIMETRES)
        rad = abs(np.radians(position))
        a = max(225*np.sin(np.linspace(0,rad,1000)))
        b = max(229.25834*np.sin(np.linspace(0,rad, 1000) + 0.195258))
        
        print(f'Distance from end a {a-79.5} and distance from end b is {b-79.5}')
        distance = 450-max(a,b)+78
        if z < distance:
            print('mouvement possible')
            self.yaw.move_absolute(position, Units.ANGLE_DEGREES)
        else:
            print('Changement de la position de z')
            self.move_z(distance-1)
            self.move_yaw(position)



    def home_rotation(self, axis):
        '''Arguments: axis
        
        Homes the rotary axis and prevents full revolution in the process'''
        position = axis.get_position(Units.ANGLE_DEGREES)
        if position < 0:
            axis.move_relative(abs(position)+1, Units.ANGLE_DEGREES)
        axis.home(False)
    
    def disconnect(self):
        self.connection1.close()

    def move_xyz(self, position, units=Units.LENGTH_MILLIMETRES):
        '''Arguments: position
        
        Moves the sliders to position x, y and z'''
        if position[0]:
            self.x_axis.move_absolute(position[0], units, False)
        if position[1]:
            self.y_axis.move_absolute(position[1], units, False)
        if position[2]:
            self.move_z(position[2], False)
        for axis in self.axis_list:
            axis.wait_until_idle()



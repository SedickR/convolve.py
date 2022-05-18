from regex import P
from zaber_motion import Library
from zaber_motion.ascii import Connection
from zaber_motion import Units
import asyncio, time
from zaber_motion.ascii import SettingConstants

Library.enable_device_db_store()



connection = Connection.open_iot_authenticated('b3baf6e3-8703-4d58-bf7f-61ecc7df8058', 'nhqcQy6_KpchnMzV6Q-9v4HCRG8zAgtB')
connection2 = Connection.open_iot_authenticated('72e1311b-c8ba-4bc8-ba4f-d3027772495f', 'nhqcQy6_KpchnMzV6Q-9v4HCRG8zAgtB')
connection3 = Connection.open_iot_authenticated('0bb78c8a-1526-42e2-954b-b91e874c859e', 'nhqcQy6_KpchnMzV6Q-9v4HCRG8zAgtB')
connection4 = Connection.open_iot_authenticated('cfda2586-ce03-445e-8577-587d68e2d06a', 'nhqcQy6_KpchnMzV6Q-9v4HCRG8zAgtB')

test = connection.detect_devices()[0].get_axis(1)
test2 = connection2.detect_devices()[0].get_axis(1)
test3 = connection3.detect_devices()[0].get_axis(1)
test4 = connection4.detect_devices()[0].get_axis(1)

axis_list = [test2, test4]
rotary_list = [test, test3]
all_axes = [test, test2, test3, test4]

for axis in axis_list:
    axis.move_absolute(axis.settings.get(SettingConstants.LIMIT_MAX))

for axis in rotary_list:
    axis.move_absolute(90, Units.ANGLE_DEGREES)




def home_rotation(axis):
    position = axis.get_position(Units.ANGLE_DEGREES)
    if position < 0:
        axis.move_relative(abs(position)+1, Units.ANGLE_DEGREES, False)
        axis.wait_until_idle()
    axis.home(False)


for axis in rotary_list:
    home_rotation(axis)
for axis in axis_list:
    axis.home(False)
for axis in all_axes:
    axis.wait_until_idle()



connection.close()
connection2.close()
connection3.close()
connection4.close()
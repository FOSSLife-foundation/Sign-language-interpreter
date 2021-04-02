from random import random, uniform
from math import sin, cos

def find_center(point_list):
    tot_x = 0
    tot_y = 0
    n = len(point_list)
    for i in point_list:
        tot_x += i[0]
        tot_y += i[1]
    return [tot_x/n, tot_y/n]

def flip_keypoints(point_list):
    flipped_list = [[1 - i[0], i[1]] for i in point_list]
    return flipped_list

def scale_keypoints(point_list, min_scale = 0.8, max_scale = 1.6):
    rand_scale = 1 + (max_scale - min_scale) * (random() - (1 - min_scale))
    center = find_center(point_list)
    scaled_list = [[center[0] + (i[0] - center[0]) * rand_scale, center[1] + (i[1] - center[1]) * rand_scale] for i in point_list]
    return scaled_list

def move_keypoints(point_list):
    min_x = min(point_list, key=lambda x: x[0])[0]
    max_x = max(point_list, key=lambda x: x[0])[0]
    min_y = min(point_list, key=lambda x: x[1])[1]
    max_y = max(point_list, key=lambda x: x[1])[1]
    
    disp = [uniform(-min_x, 1 - max_x), uniform(-min_y, 1 - max_y)]
    moved_list = []
    for point in point_list:
        moved_list.append([point[0] + disp[0], point[1] + disp[1]])
    
    return moved_list

def rotate_point(center, p, angle):
    point = [p[0], p[1]]
    s = sin(angle)
    c = cos(angle)
    
    point[0] -= center[0]
    point[1] -= center[1]
    
    xnew = point[0] * c - point[1] * s
    ynew = point[0] * s + point[1] * c
    
    point[0] = xnew + center[0]
    point[1] = ynew + center[1]
    
    return point

def rotate_point(center, p, s, c):
    point = [p[0] * 1280, p[1] * 720]
    
    point[0] -= center[0]
    point[1] -= center[1]
    
    xnew = point[0] * c - point[1] * s
    ynew = point[0] * s + point[1] * c
    
    point[0] = (xnew + center[0])/1280
    point[1] = (ynew + center[1])/720
    
    return point

def rotate_key_points(point_list, max_angle_in_rad = 0.5):
    angle = random() * max_angle_in_rad
    center = find_center(point_list)
    center = [center[0] * 1280, center[1] * 720]
    s = sin(angle)
    c = cos(angle)
    rotated_list = []
    for point in point_list:
        rotated_list.append(rotate_point(center, point, s, c))
    
    return rotated_list
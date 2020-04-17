# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:20:53 2020

@author: kavic
"""
import math

def distance(x,y):
    x1, y1 = x
    x2, y2 = y
    distance_travelled = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance_travelled

def time(frame1,frame2,fps):
    time_travelled = (frame2 - frame1)/fps
    return time_travelled

def speed(distance,time):
    speed = distance / time 
    return speed

def cal_remaining_time (x, frame, fps):
    remaining_time = x * (frame / fps)
    return remaining_time
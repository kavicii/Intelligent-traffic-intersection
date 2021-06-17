# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:22:27 2020

@author: kavic
"""
import cv2

def get_traffic_light_status(time):
    round_time = round(time)%70
    veh_light = 0
    ped_light = 0
    light_remaining_time = 35 - round_time
    if round_time > 35 and round_time <= 39:
        veh_light = 1
        ped_light = 0
        light_remaining_time = 39 - round_time
    if round_time > 39 and round_time <= 42:
        veh_light = 2
        ped_light = 0
        light_remaining_time = 42 - round_time
    if round_time > 42 and round_time <= 54:
        veh_light = 2
        ped_light = 1
        light_remaining_time = 54 - round_time
    if round_time > 54 and round_time <= 66:
        veh_light = 2
        ped_light = 2
        light_remaining_time = 66 - round_time
    if round_time > 66 and round_time <= 69:
        veh_light = 3
        ped_light = 0
        light_remaining_time = 69 - round_time
    return veh_light, ped_light , light_remaining_time

# =============================================================================
# def adjust_time(start_time, extend_flag, reduce_flag):
#     if extend_flag:
#         start_time - 6
#         extend_flag = False
#         print('extended')
#     if reduce_flag:
#         start_time + 6
#         reduce_flag = False
#         print('reducing')
#     return start_time , extend_flag , reduce_flag
# =============================================================================
    
def get_traffic_light_image(frame, detector, veh_light, ped_light):
    if detector == 'yolo':
        if veh_light == 0:
            img = cv2.imread('./data/traffic_light/veh_light_0.jpg')
        if veh_light == 1:
            img = cv2.imread('./data/traffic_light/veh_light_1.jpg')
        if veh_light == 2:
            img = cv2.imread('./data/traffic_light/veh_light_2.jpg')
        if veh_light == 3:
            img = cv2.imread('./data/traffic_light/veh_light_3.jpg')
    if detector == 'yolo_p':
        if ped_light == 0:
            img = cv2.imread('./data/traffic_light/ped_light_0.jpg')
        if ped_light == 1:
            img = cv2.imread('./data/traffic_light/ped_light_1.jpg')
        if ped_light == 2:
            img = cv2.imread('./data/traffic_light/ped_light_2.jpg')
    scale_percent = 300       # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    rows,cols,channels = resized_img.shape
    roi = frame[0:rows, 0:cols ]
    
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    # Now black-out the area of logo in ROI
    frame_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    
    # Take only region of logo from logo image.
    img_fg = cv2.bitwise_and(resized_img,resized_img,mask = mask)
    
    # Put logo in ROI and modify the main image
    dst = cv2.add(frame_bg,img_fg)
    frame[0:rows, 0:cols ] = dst
    return frame
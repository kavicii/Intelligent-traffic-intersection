# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:13:35 2020

@author: kavic
"""
import cv2 

def get_text_box_with_background_color(frame,text,coords,font,font_scale,bgr,thickness,line_type,bg_bgr):
    (text_width, text_height) = cv2.getTextSize(text, font, font_scale, thickness)[0]
    box_coords = ((coords[0],coords[1]-25),(coords[0]+text_width,coords[1]+text_height-15))
    cv2.rectangle(frame, box_coords[0], box_coords[1], bg_bgr, cv2.FILLED)
    cv2.putText(frame, text, coords, font,font_scale, bgr, thickness,line_type)
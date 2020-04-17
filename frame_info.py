# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:00:00 2020

@author: kavic
"""
import cv2
from util.logger import get_logger
logger = get_logger()
def get_frame_info(cap, _timer):
        processing_frame_rate = round(cv2.getTickFrequency() / (cv2.getTickCount() - _timer), 2)
        frames_processed = round(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frames_count = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fram_rate = processing_frame_rate
        logger.debug('Frame processed.', extra={
            'meta': {
                'label': 'FRAME_PROCESS',
                'frames_processed': frames_processed,
                'frame_rate': processing_frame_rate,
                'frames_left': frames_count - frames_processed,
                'percentage_processed': round((frames_processed / frames_count) * 100, 2),
            },
        }) 
def get_frame_rate:
    return frame_rate
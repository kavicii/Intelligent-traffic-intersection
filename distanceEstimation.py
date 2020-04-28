# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 00:07:29 2020

@author: kavic
"""
def UpdateNearestBlobPosition(blob,blob_id, blobs_list, nearestBlobPos, nearestBlob, rescan_requested):
    
    x, y, w, h = blob.bounding_box
    if y+h > nearestBlobPos[1]:
        nearestBlob = blob_id, blob
        nearestBlobPos = (round((x + x + w) / 2),y+h)
    if nearestBlob not in blobs_list:
        rescan_requested = True
# =============================================================================
#         nearestBlob = blob, blob_id
#         nearestBlobPos = (round((x + x + w) / 2),y+h)
# =============================================================================
    return nearestBlob , nearestBlobPos , rescan_requested

# =============================================================================
# def checkNearestBlobExit(blob,blob_id, blobs_list,nearestBlobPos, nearestBlob):
#     if nearestBlob not in blobs_list:
#         nearestBlob = blob, blob_id
#     return nearestBlob , nearestBlobPos
# =============================================================================


def distance_cal(objectPosition,cameraHeight,focalLength,pixelLength,resolution,vanishingPoint):
# =============================================================================
#     cameraHeight = 4.5
#     focalLength = 3.6 * (10**-3)
#     pixelLength = 4.8 * (10**-3) / 1920
#     vanishingPoint = 246
# =============================================================================
    estimateDistance = cameraHeight * focalLength / ((pixelLength/resolution[1])*(objectPosition - vanishingPoint[1]))
    return estimateDistance

def normalizing_1frame_distance(x,y,nearestBlob1FrameDistance):
    if len(nearestBlob1FrameDistance)<=15 :
        nearestBlob1FrameDistance.append(abs(x-y))
    else:
        nearestBlob1FrameDistance.pop(0)
        nearestBlob1FrameDistance.append(abs(x-y))
    normalized1FrameDistance = sum(nearestBlob1FrameDistance)/len(nearestBlob1FrameDistance)
    return normalized1FrameDistance
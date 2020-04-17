
"""
Created on Tue Apr 14 03:40:47 2020

@author: kavic
"""


def _is_on_which_roads (roads, blob):
    dif = []
    roadXCoordinates = []
    x, y, w, h = blob.bounding_box
    for road in roads:
        roadXCoordinates.append(getcoordinate(y+h, road))
    
    for roadXCoordinate in roadXCoordinates:
        dif.append(abs(round((x + x + w) / 2) - roadXCoordinate))
    i = dif.index(min(dif))
    return roads[i]["label"]


def getcoordinate(yCoordinate, road):
    x1 = road["line"][0][0]
    y1 = road["line"][0][1]
    x2 = road["line"][1][0]
    y2 = road["line"][1][1]
    xCoordinate = ((yCoordinate - y1) * (x2 - x1) / (y2 - y1)) + x1
    return xCoordinate
    
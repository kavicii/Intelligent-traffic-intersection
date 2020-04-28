from .bounding_box import get_centroid, get_area, get_position, get_bottom_point
from distanceEstimation import distance_cal

class Blob:
    '''
    A blob object represents a tracked vehicle as it moves around in a video.
    '''
    def __init__(self, _bounding_box, _type, _confidence, _tracker):
        self.bounding_box = _bounding_box
        self.type = _type
        self.type_confidence = _confidence
        self.centroid = get_centroid(_bounding_box)
        self.bottom_point = get_bottom_point(_bounding_box)
        self.area = get_area(_bounding_box)
        self.tracker = _tracker
        self.num_consecutive_tracking_failures = 0
        self.num_consecutive_detection_failures = 0
        self.lines_crossed = [] # list of counting lines crossed by a vehicle
        self.speed_estimation_line_crossed = []
        self.lines_crossed_frame = 1
        # self.position_first_detected = tuple(self.centroid)
        # self.centroid_crossed_line = (0,0)
        self.speed = 0
        self.onRoad = "A"
        self.position = get_position(_bounding_box)
        self.distance = 0
        self.distanceInLast3Frame = []
        self.distanceBetween3Frames = []
        self.mean_distanceInLast3Frame= 0.0
        self.remainingTime = 0.0
        self.isJaywalker = False
        
        
        
    def update(self, _bounding_box, _type=None, _confidence=None, _tracker=None):
        self.bounding_box = _bounding_box
        self.type = _type if _type is not None else self.type
        self.type_confidence = _confidence if _confidence is not None else self.type_confidence
        self.centroid = get_centroid(_bounding_box)
        self.area = get_area(_bounding_box)
        self.position = get_position(_bounding_box)
        # self.distance = distance_cal(self.position)
        if _tracker:
            self.tracker = _tracker
    

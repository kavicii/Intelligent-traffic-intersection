'''
Vehicle Counter class.
'''

# pylint: disable=missing-class-docstring,missing-function-docstring,invalid-name

import multiprocessing
import cv2
from joblib import Parallel, delayed

from tracker import add_new_blobs, remove_duplicates, update_blob_tracker
from detectors.detector import get_bounding_boxes
from util.detection_roi import get_roi_frame, draw_roi
from util.logger import get_logger
from counter import attempt_count
from distanceEstimation import UpdateNearestBlobPosition, distance_cal, normalizing_1frame_distance
from speed import cal_remaining_time
from roads import _is_on_which_roads
logger = get_logger()
NUM_CORES = multiprocessing.cpu_count()

class VehicleCounter():

    def __init__(self, initial_frame, detector, tracker, droi, show_droi, mcdf, mctf, di, counting_lines, speed_estimation_lines , show_counts,frames_processed, processing_frame_rate, roads):
        self.frame = initial_frame # current frame of video
        self.detector = detector
        self.tracker = tracker
        self.droi = droi # detection region of interest
        self.show_droi = show_droi
        self.mcdf = mcdf # maximum consecutive detection failures
        self.mctf = mctf # maximum consecutive tracking failures
        self.detection_interval = di
        self.counting_lines = counting_lines
        self.roads = roads
        self.speed_estimation_lines = speed_estimation_lines

        self.blobs = {}
        self.f_height, self.f_width, _ = self.frame.shape
        self.frame_count = 0 # number of frames since last detection
        self.counts = {counting_line['label']: {} for counting_line in counting_lines} # counts of vehicles by type for each counting line
        self.show_counts = show_counts
        self.processing_frame_rate = processing_frame_rate
        self.frames_processed = frames_processed
        self.nearestBlobPosition = (0.0,0.0)
        self.nearestBlobLastFramePosition = (0.0,0.0)
        self.nearestBlobDistance = 0.0
        self.nearestBlobLastFrameDistance=0.0
        self.nearestBlob1FrameDistance=[]
        self.NearestBlob = []
        self.NearestBlob_remaining_time = 0.0
        
        # create blobs from initial frame
        droi_frame = get_roi_frame(self.frame, self.droi)
        _bounding_boxes, _classes, _confidences = get_bounding_boxes(droi_frame, self.detector)
        self.blobs = add_new_blobs(_bounding_boxes, _classes, _confidences, self.blobs, self.frame, self.tracker, self.mcdf)

    def get_blobs(self):
        return self.blobs

    def count(self, frame,frames_processed, processing_frame_rate):
        self.frame = frame
        self.processing_frame_rate = processing_frame_rate
        self.frames_processed = frames_processed
        rescan_requested = False
        self.nearestBlobLastFrameDistance = self.nearestBlobDistance
        blobs_list = list(self.blobs.items())
        # update blob trackers
        blobs_list = Parallel(n_jobs=NUM_CORES, prefer='threads')(
            delayed(update_blob_tracker)(blob, blob_id, self.frame) for blob_id, blob in blobs_list
        )
        self.blobs = dict(blobs_list)

        for blob_id, blob in blobs_list:
            # count vehicle if it has crossed a counting line
            blob, self.counts = attempt_count(blob, blob_id, self.counting_lines,self.speed_estimation_lines , self.counts, self.frames_processed,self.processing_frame_rate)
            self.blobs[blob_id] = blob

            # remove blob if it has reached the limit for tracking failures
            if blob.num_consecutive_tracking_failures >= self.mctf:
                del self.blobs[blob_id]
            blob.onRoad =  str(_is_on_which_roads(self.roads, blob))
            self.NearestBlob, self.nearestBlobPosition , rescan_requested = UpdateNearestBlobPosition(blob, blob_id, blobs_list, self.nearestBlobPosition, self.NearestBlob, rescan_requested)
            self.nearestBlobDistance = distance_cal(self.nearestBlobPosition[1])           

        if rescan_requested:
            self.nearestBlob1FrameDistance=[]
            self.nearestBlobPosition = (0,0)
            for blob_id, blob in blobs_list:
                 self.NearestBlob, self.nearestBlobPosition , rescan_requested = UpdateNearestBlobPosition(blob, blob_id, blobs_list, self.nearestBlobPosition, self.NearestBlob,rescan_requested)
                 self.nearestBlobDistance = distance_cal(self.nearestBlobPosition[1])
            self.nearestBlobLastFrameDistance = self.nearestBlobDistance
            rescan_requested = False

        if self.processing_frame_rate > 0 and self.nearestBlobPosition[1] > 0 and self.nearestBlobDistance != self.nearestBlobLastFrameDistance and rescan_requested == False:
# =============================================================================
#             print('nor_list'+str(normalizing_1frame_distance(self.nearestBlobDistance,self.nearestBlobLastFrameDistance,self.nearestBlob1FrameDistance)))
#             print('list'+str(self.nearestBlob1FrameDistance))
#             print('1  '+str(self.nearestBlobDistance))
#             print('2  '+str(self.nearestBlobLastFrameDistance))
# =============================================================================
            self.NearestBlob_remaining_time = cal_remaining_time( (self.nearestBlobDistance / normalizing_1frame_distance(self.nearestBlobDistance,self.nearestBlobLastFrameDistance,self.nearestBlob1FrameDistance)),1,self.processing_frame_rate)
        

            
        if self.frame_count >= self.detection_interval:
            # rerun detection
            droi_frame = get_roi_frame(self.frame, self.droi)
            _bounding_boxes, _classes, _confidences = get_bounding_boxes(droi_frame, self.detector)

            self.blobs = add_new_blobs(_bounding_boxes, _classes, _confidences, self.blobs, self.frame, self.tracker, self.mcdf)
            self.blobs = remove_duplicates(self.blobs)
            self.frame_count = 0

        self.frame_count += 1
        

    def visualize(self, roi_setting_mode,is_paused,showUI):
        frame = self.frame
        font = cv2.FONT_HERSHEY_TRIPLEX
        line_type = cv2.LINE_AA
        cv2.putText(frame, 'fps: ' +str(self.processing_frame_rate), (self.f_width - 150 , 50), font, 1, (255, 255, 0), 1, line_type)
        
        # draw and label blob bounding boxes
        for _id, blob in self.blobs.items():
            (x, y, w, h) = [int(v) for v in blob.bounding_box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            vehicle_label = 'I: ' + _id[:8] + ' on Road '+ blob.onRoad \
                        if blob.speed == 0 \
                        else 'ID: {0}, Speed: {1} '.format(_id[:4], round( blob.speed))
                            # if blob.type is None \
                            # else 'I: {0}, T: {1} ({2})'.format(_id[:8], blob.type, str(blob.type_confidence)[:4])
            cv2.putText(frame, vehicle_label, (x, y - 5), font, 1, (0, 0, 0), 1, line_type)

        # draw counting lines
        for counting_line in self.counting_lines:
            cv2.line(frame, counting_line['line'][0], counting_line['line'][1], (255, 0, 0), 3)
            cl_label_origin = (counting_line['line'][0][0], counting_line['line'][0][1] + 35)
            cv2.putText(frame, str(round(self.nearestBlobDistance,1))+'m ('+str(round(self.NearestBlob_remaining_time,1)) + 's)' , cl_label_origin, font, 1, (255, 0, 0), 2, line_type)
        
# =============================================================================
#         for road in self.roads:
#             cv2.line(frame, road['line'][0], road['line'][1], (255, 0, 0), 3)
#             r_label_origin = (road['line'][1][0] - 15 , road['line'][1][1] - 35)
#             cv2.putText(frame, road['label'] , r_label_origin, font, 1, (255, 0, 0), 2, line_type)
# =============================================================================
# =============================================================================
#         for speed_estimation_line in self.speed_estimation_lines:
#             cv2.line(frame, speed_estimation_line['line'][0], speed_estimation_line['line'][1], (255, 0, 0), 3)
#             sel_label_origin = (speed_estimation_line['line'][0][0], speed_estimation_line['line'][0][1] + 35)
#             cv2.putText(frame, speed_estimation_line['label']+" speed", sel_label_origin, font, 1, (255, 0, 0), 2, line_type)
#         
# =============================================================================

# =============================================================================
#         if is_paused:
#             cv2.putText(frame, 'Pausing', (round(self.f_width/2),30),  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
# =============================================================================

        if roi_setting_mode:
            cv2.putText(frame, 'Setting ROI', (round(self.f_width/2),30),  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        
        # show detection roi
        if self.show_droi:
            frame = draw_roi(frame, self.droi)
        
        if showUI:
            for road in self.roads:
                cv2.line(frame, road['line'][0], road['line'][1], (255, 0, 0), 3)
                r_label_origin = (road['line'][1][0] - 15 , road['line'][1][1] - 35)
                cv2.putText(frame, road['label'] , r_label_origin, font, 1, (255, 0, 0), 2, line_type)
        # show counts
        if self.show_counts:
            offset = 1
            for line, objects in self.counts.items():
                cv2.putText(frame, line, (10, 40 * offset), font, 1, (255, 0, 0), 2, line_type)
                for label, count in objects.items():
                    offset += 1
                    cv2.putText(frame, "{}: {}".format(label, count), (10, 40 * offset), font, 1, (255, 0, 0), 2, line_type)
                offset += 2

        return frame

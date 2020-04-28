'''
Vehicle Counter class.
'''

# pylint: disable=missing-class-docstring,missing-function-docstring,invalid-name

import multiprocessing
import cv2
import time
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from joblib import Parallel, delayed

from tracker import add_new_blobs, remove_duplicates, update_blob_tracker
from detectors.detector import get_bounding_boxes
from counter import attempt_count, create_waiting_zone
from distanceEstimation import UpdateNearestBlobPosition, distance_cal, normalizing_1frame_distance
from speed import cal_remaining_time
from roads import _is_on_which_roads
from util.detection_roi import get_roi_frame, draw_roi
from util.logger import get_logger
from util.text import get_text_box_with_background_color
from util.equalization import list_equalization
from util.bounding_box import get_position
from traffic_light import get_traffic_light_image

logger = get_logger()
NUM_CORES = multiprocessing.cpu_count()

class Processor():

    def __init__(self, initial_frame, detector, tracker, droi, show_droi, mcdf, mctf, di, counting_lines, speed_estimation_lines , show_counts,frames_processed, processing_frame_rate, roads,veh_light, ped_light, camera_height, focal_length, pixel_length, resolution, vanishing_point):
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
        self.blobID = 0
        self.blobDistance = 0.0
        self.f_height, self.f_width, _ = self.frame.shape
        self.frame_count = 0 # number of frames since last detection
        self.counts = {counting_line['label']: {} for counting_line in counting_lines} # counts of vehicles by type for each counting line
        self.show_counts = show_counts
        self.processing_frame_rate = processing_frame_rate
        self.frames_processed = frames_processed
        self.waiting_zone = create_waiting_zone(self.counting_lines,self.droi)
        self.isWaiting = 0
        self.veh_light = veh_light
        self.ped_light = ped_light
        self.extend_flag = False
        self.reduce_flag = False
        self.extend_notification = 0
        self.reduce_notification = 0
        self.jaywalker_flag = False
        self.not_slowing_down_flag = False
        
        self.camera_height = camera_height
        self.focal_length = focal_length
        self.pixel_length= pixel_length
        self.resolution = resolution
        self.vanishing_point = vanishing_point
# =============================================================================
#         self.nearestBlobPosition = (0.0,0.0)
#         self.nearestBlobLastFramePosition = (0.0,0.0)
#         self.nearestBlobDistance = 0.0
#         self.nearestBlobLastFrameDistance=0.0
#         self.nearestBlob1FrameDistance=[]
#         self.NearestBlob = []
#         self.NearestBlob_remaining_time = 0.0
# =============================================================================
        
        # create blobs from initial frame((3) times)
        for x in range(6):
            self.blobID += 1
            droi_frame = get_roi_frame(self.frame, self.droi)
            _bounding_boxes, _classes, _confidences = get_bounding_boxes(droi_frame, self.detector)
            self.blobs = add_new_blobs(_bounding_boxes, _classes, _confidences, self.blobs, self.frame, self.tracker, self.mcdf, self.blobID)

    def get_blobs(self):
        return self.blobs
    
    def get_extend_flag(self):
        return self.extend_flag
    
    def get_reduce_flag(self):
        return self.reduce_flag
    
    def clear_flag(self):
        self.extend_flag = False
        self.reduce_flag = False
        
    def count(self, frame,frames_processed, processing_frame_rate):
        self.frame = frame
        self.processing_frame_rate = processing_frame_rate
        self.frames_processed = frames_processed
        self.isWaiting = 0
        # rescan_requested = False
        # self.nearestBlobLastFrameDistance = self.nearestBlobDistance
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
            if self.detector == 'yolo':
                blob.onRoad =  str(_is_on_which_roads(self.roads, blob))
                blob.distance = distance_cal(blob.position,self.camera_height,self.focal_length,self.pixel_length,self.resolution,self.vanishing_point)
                # self.NearestBlob, self.nearestBlobPosition , rescan_requested = UpdateNearestBlobPosition(blob, blob_id, blobs_list, self.nearestBlobPosition, self.NearestBlob, rescan_requested)
                # self.nearestBlobDistance = distance_cal(self.nearestBlobPosition[1])           
            # check blob is waiting
            if self.detector == 'yolo_p':
                if self.veh_light == 0:
                    point = Point(blob.bottom_point)
                    polygon = Polygon(self.waiting_zone[0:4])
                    polygon2 = Polygon(self.waiting_zone[-4:])
                    if polygon.contains(point) == True or polygon2.contains(point) == True:
                        blob.isJaywalker = False
                        self.isWaiting += 1
                    else:
                        blob.isJaywalker = True
                        self.jaywalker_flag = True
                if self.ped_light != 0:
                     blob.isJaywalker = False
                     self.jaywalker_flag = False
                    
# =============================================================================
#         if rescan_requested:
#             self.nearestBlob1FrameDistance=[]
#             self.nearestBlobPosition = (0,0)
#             for blob_id, blob in blobs_list:
#                  self.NearestBlob, self.nearestBlobPosition , rescan_requested = UpdateNearestBlobPosition(blob, blob_id, blobs_list, self.nearestBlobPosition, self.NearestBlob,rescan_requested)
#                  self.nearestBlobDistance = distance_cal(self.nearestBlobPosition[1])
#             self.nearestBlobLastFrameDistance = self.nearestBlobDistance
#             rescan_requested = False
# =============================================================================

        # if self.processing_frame_rate > 0 and self.nearestBlobPosition[1] > 0 and self.nearestBlobDistance != self.nearestBlobLastFrameDistance and rescan_requested == False:
# =============================================================================
#             print('nor_list'+str(normalizing_1frame_distance(self.nearestBlobDistance,self.nearestBlobLastFrameDistance,self.nearestBlob1FrameDistance)))
#             print('list'+str(self.nearestBlob1FrameDistance))
#             print('1  '+str(self.nearestBlobDistance))
#             print('2  '+str(self.nearestBlobLastFrameDistance))
# =============================================================================
            # self.NearestBlob_remaining_time = cal_remaining_time( (self.nearestBlobDistance / normalizing_1frame_distance(self.nearestBlobDistance,self.nearestBlobLastFrameDistance,self.nearestBlob1FrameDistance)),1,self.processing_frame_rate)
        

            
        if self.frame_count >= self.detection_interval:
            # rerun detection
            droi_frame = get_roi_frame(self.frame, self.droi)
            _bounding_boxes, _classes, _confidences = get_bounding_boxes(droi_frame, self.detector)
            self.blobID += 1
            self.blobs = add_new_blobs(_bounding_boxes, _classes, _confidences, self.blobs, self.frame, self.tracker, self.mcdf, self.blobID)
            self.blobs = remove_duplicates(self.blobs)
            self.frame_count = 0

        self.frame_count += 1
        

    def visualize(self, roi_setting_mode, is_paused, showUI, veh_light,ped_light,light_time, roi_setting_mode_P,
                  line_setting_mode,line_setting_mode_P,vpoint_setting_mode,help_img_flag):
        frame = self.frame
        font = cv2. FONT_HERSHEY_COMPLEX
        line_type = cv2.LINE_AA
        occupys = []
        offset = 0
        lineDistance = distance_cal(self.counting_lines[0]['line'][0][1],self.camera_height,self.focal_length,self.pixel_length,self.resolution,self.vanishing_point)
        text1 = 'fps: ' +str(round(self.processing_frame_rate,2))
        text2 = 'Now have: ' +str(len(self.blobs))
        cv2.putText(frame, str(time.asctime( time.localtime(time.time()))), (0 , self.f_height-10), font, 1, (255, 255, 0), 1, line_type)
        get_text_box_with_background_color(frame,text1,(self.f_width - 300 , 50),font,1,(255, 255, 255),1,line_type,(0,0,0))
        get_text_box_with_background_color(frame,text2,(self.f_width - 300 , 100),font,1,(255, 255, 255),1,line_type,(0,0,0))
        # cv2.putText(frame, 'fps: ' +str(self.processing_frame_rate), (self.f_width - 150 , 50), font, 1, (25, 25, 25), 1, line_type)
        # cv2.putText(frame, 'Now have: ' +str(len(self.blobs))+' objects', (self.f_width - 350 , 100), font, 1, (25, 25, 25), 1, line_type) 
        # draw and label blob bounding boxes
        self.veh_light = veh_light
        self.ped_light = ped_light
        frame = get_traffic_light_image(frame, self.detector, veh_light, ped_light)
        get_text_box_with_background_color(frame,str(light_time),(10 , 400),font,1,(255, 255, 255),1,line_type,(0,0,0))
        for _id, blob in self.blobs.items():
            (x, y, w, h) = [int(v) for v in blob.bounding_box]
            if blob.isJaywalker == True:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            for occupy in occupys:
                if y >= occupy[0] and y <= occupy[1]:
                    offset+=1
                if y-31 > occupy[0] and y-31 < occupy [1]:
                    offset+=1
            # cv2.putText(frame, 'ID:' + str(_id), (x , y - 25), font, 0.7, (0, 0, 0), 2, line_type)
            if self.detector == 'yolo':
                distanceToLine = blob.distance - lineDistance
                blob.distanceInLast3Frame.append(distanceToLine)
                if len(blob.distanceInLast3Frame) > 3:
                    blob.distanceInLast3Frame.pop(0)           
                if len(blob.distanceInLast3Frame) == 3:
                    blob.mean_distanceInLast3Frame = list_equalization(blob.distanceBetween3Frames,abs(blob.distanceInLast3Frame[0] - blob.distanceInLast3Frame[2]),15)
                    blob.remainingTime = cal_remaining_time(round(distanceToLine-blob.mean_distanceInLast3Frame,1),2,self.processing_frame_rate)
                    if round(blob.remainingTime,1) > 0:
                        blob.speed = round(distanceToLine/blob.remainingTime,1)
                        get_text_box_with_background_color(frame,str(round(distanceToLine,1)) +'m '+ str(round(blob.remainingTime,1))+'s',(x , y - 5-(offset*35)),font,0.7,(0, 0, 0),2,line_type,(223,223,223))
                        if not self.veh_light == 0:
                            if blob.speed >= 1.5 and distanceToLine <= 40:
                                self.not_slowing_down_flag = True
                            else:
                                self.not_slowing_down_flag = False
                        else:
                                self.not_slowing_down_flag = False
                        # +str(round(distanceToLine/blob.remainingTime*3.6,1))+' km/h'
                        # cv2.putText(frame, str(round(distanceToLine,1)) +'m'+ str(round(blob.remainingTime,1))+'s', (x , y - 5), font, 0.7, (0, 0, 0), 2, line_type)
                else:
                    if round(distanceToLine,1) > 0:
                        get_text_box_with_background_color(frame,str(round(distanceToLine,1)) +'m ',(x , y - 5-(offset*35)),font,0.7,(0, 0, 0),2,line_type,(223,223,223))
                        # cv2.putText(frame, str(round(distanceToLine,1)) +'m' , (x , y - 5), font, 0.7, (0, 0, 0), 2, line_type)
                get_text_box_with_background_color(frame,'ID:' + str(_id),(x , y - 31 - (offset*35)),font,0.7,(0, 0, 0),2,line_type,(223,223,223))
                cv2.line(frame, (x , y - 5-(offset*35)), (x,y), (255, 0, 0), 3)
                occupys.append((y - 31 - (offset*35), y - (offset*35)))                               
            else:
                get_text_box_with_background_color(frame,'ID:' + str(_id),(x , y - 5),font,0.7,(0, 0, 0),2,line_type,(223,223,223))
                if blob.isJaywalker:
                    get_text_box_with_background_color(frame,"Jaywalker detected.",(x , y - 30),font,0.7,(0, 0, 0),2,line_type,(223,223,223))
               
        # draw counting lines
        for counting_line in self.counting_lines:
            cv2.line(frame, counting_line['line'][0], counting_line['line'][1], (255, 0, 0), 3)
            # cl_label_origin = (counting_line['line'][0][0], counting_line['line'][0][1] + 35)
            # cv2.putText(frame, str(round(self.blobDistance,1))+'m (' , cl_label_origin, font, 1, (255, 0, 0), 2, line_type)
        
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

        if roi_setting_mode and self.detector == 'yolo':
            get_text_box_with_background_color(frame,'Setting ROI.',(round(self.f_width/2)-150,30),font,1,(0, 0, 0),2,line_type,(255,255,255))
        if roi_setting_mode_P and self.detector == 'yolo_p':        
            get_text_box_with_background_color(frame,'Setting ROI.',(round(self.f_width/2)-150,30),font,1,(0, 0, 0),2,line_type,(255,255,255))
        if line_setting_mode and self.detector == 'yolo':
            get_text_box_with_background_color(frame,'Setting Line.',(round(self.f_width/2)-150,30),font,1,(0, 0, 0),2,line_type,(255,255,255))
        if line_setting_mode_P and self.detector == 'yolo_p':
            get_text_box_with_background_color(frame,'Setting Line.',(round(self.f_width/2)-150,30),font,1,(0, 0, 0),2,line_type,(255,255,255))
        if vpoint_setting_mode and self.detector == 'yolo':
            get_text_box_with_background_color(frame,'Setting Vanishing Point.',(round(self.f_width/2)-150,30),font,1,(0, 0, 0),2,line_type,(255,255,255))        
        if help_img_flag and self.detector == 'yolo':
            cv2.putText(frame, "Press 'h' to close user manual.", (self.f_width-750,self.f_height-30),  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        if help_img_flag is False and self.detector == 'yolo':   
             cv2.putText(frame, "Press 'h' to open user manual", (self.f_width-750,self.f_height-30),  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        # show detection roi
        if self.show_droi:
            frame = draw_roi(frame, self.droi,(255,255,0))
            if self.detector == 'yolo_p':
                frame = draw_roi(frame,self.waiting_zone[0:4],(0,255,255))
                if len(self.waiting_zone)==8:
                    frame = draw_roi(frame,self.waiting_zone[-4:],(255,0,255))
        
        if showUI:
            if self.detector == 'yolo':
                x,y = self.vanishing_point
                cv2.line(frame, (x+50,y+50) , (x-50,y-50) , (0, 0, 0), 2)
                cv2.line(frame, (x+50,y-50) , (x-50,y+50), (0, 0, 0), 2)
                cv2.putText(frame, 'Showing Vanishing Point.', (round(self.f_width/2)-150,30),  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
# =============================================================================
#                 for road in self.roads:
#                     cv2.line(frame, road['line'][0], road['line'][1], (255, 0, 0), 3)
#                     r_label_origin = (road['line'][1][0] - 15 , road['line'][1][1] - 35)
#                     cv2.putText(frame, road['label'] , r_label_origin, font, 1, (255, 0, 0), 2, line_type)
# =============================================================================
        # show counts
        if self.show_counts:
            if self.detector == 'yolo':
                offset = 1
                for line, objects in self.counts.items():
                    cv2.putText(frame, line, (75, 40 * offset), font, 1, (255, 0, 0), 2, line_type)
                    for label, count in objects.items():
                        offset += 1
                        cv2.putText(frame, "{}: {}".format(label, count), (75, 40 * offset), font, 1, (255, 0, 0), 2, line_type)
                    offset += 2
                    
        if self.detector == 'yolo_p':
            if self.veh_light == 0:
                get_text_box_with_background_color(frame,"waiting zone have "+str(self.isWaiting)+" people.",(self.f_width - 550 , 150),font,1,(255, 255, 255),1,line_type,(0,0,0)) 
       
        # adjust traffic light duration when vehicle traffic light is red and pedestrian traffic light is green.
        if self.detector == 'yolo' and self.veh_light == 2 and self.ped_light == 1:
            if len(self.blobs) == 0 and light_time < 6:
                self.extend_flag = True
                self.extend_notification = 10
                print("2,1,process activated extend")               
        if self.detector == 'yolo_p' and self.veh_light == 2 and self.ped_light == 1:
            if len(self.blobs) == 0 and light_time > 6:
                self.reduce_flag = True
                self.reduce_notification  = 10
                print("2,1,process activated reduce") 
        if self.extend_notification != 0 :
            cv2.putText(frame, 'traffic light duration is extended', (10 , 450),  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            self.extend_notification -= 0
        if self.reduce_notification != 0 :
            cv2.putText(frame, 'traffic light duration is reduced', (10 , 450),  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            self.reduce_notification -= 0 
            
        # adjust traffic light duration when vehicle traffic light is green and pedestrian traffic light is red.
        if self.detector == 'yolo' and self.veh_light == 0 and self.ped_light == 0:
            # print(light_time)
            # print(self.veh_light)
            # print(self.ped_light)
            if len(self.blobs) == 0  and light_time > 6:
                self.reduce_flag = True
                self.reduce_notification  = 10   
                print("0,0,process activated reduce") 
        if self.detector == 'yolo_p' and self.veh_light == 0 and self.ped_light == 0:
            if self.isWaiting == 0 and light_time < 6:
                self.extend_flag = True
                self.extend_notification = 10
                print("0,0,process activated extend")     
        if self.extend_notification != 0 :
            cv2.putText(frame, 'traffic light duration is extended', (10 , 450),  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            self.extend_notification -= 1
        if self.reduce_notification != 0 :
            cv2.putText(frame, 'traffic light duration is reduced', (10 , 450),  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            self.reduce_notification -= 1 
        if self.jaywalker_flag:
            get_text_box_with_background_color(frame,"Warning: Jaywalker detected.",(250 , 100),font,2,(255, 255, 255),2,line_type,(0,0,255))
            cv2.rectangle(frame, (0, 0), (self.f_width, self.f_height), (0, 0, 255), 10)
        if self.not_slowing_down_flag:
           get_text_box_with_background_color(frame,"Warning: High Speed Vehicle deteced.",(250 , 100),font,1,(255, 255, 255),2,line_type,(0,0,255))
           cv2.rectangle(frame, (0, 0), (self.f_width, self.f_height), (0, 0, 255), 10) 
        return frame

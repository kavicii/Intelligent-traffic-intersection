'''
VCS entry point.
'''

# pylint: disable=wrong-import-position

import time
import cv2

from dotenv import load_dotenv
load_dotenv()
import settings
from util.logger import init_logger
from util.image import take_screenshot
from util.logger import get_logger
from util.equalization import list_equalization
from Processor import Processor
from util.debugger import CoordinateStore
from traffic_light import get_traffic_light_status
init_logger()
logger = get_logger()
import sys
import os

def run():
    '''
    Initialize counter class and run counting loop.
    '''

    # capture traffic scene video
    is_cam = settings.IS_CAM
    video = settings.VIDEO
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise Exception('Invalid video source {0}'.format(video))
    ret, frame = cap.read()
    f_height, f_width, _ = frame.shape

    #capture pedestrians video
    is_cam_P = settings.IS_CAM_P
    video_P = settings.VIDEO_P
    cap_P = cv2.VideoCapture(video_P)
    if not cap_P.isOpened():
        raise Exception('Invalid video source {0}'.format(video_P))
    ret_P, frame_P = cap_P.read()
    f_P_height, f_P_width, _ = frame_P.shape
    detection_interval = settings.DI
    mcdf = settings.MCDF
    mctf = settings.MCTF
    detector = settings.DETECTOR
    detectorP = settings.DETECTOR_P
    tracker = settings.TRACKER
    # create detection region of interest polygon
    use_droi = settings.USE_DROI
    use_droi_P = settings.USE_DROI_P
    droi = settings.DROI \
            if use_droi \
            else [(0, 0), (f_width, 0), (f_width, f_height), (0, f_height)]
    droi_P = settings.DROI_P \
            if use_droi_P \
            else [(0, 0), (f_P_width, 0), (f_P_width, f_P_height), (0, f_P_height)]
    show_droi = settings.SHOW_DROI
    show_droi_P = settings.SHOW_DROI_P
    counting_lines = settings.COUNTING_LINES
    counting_lines_P = settings.COUNTING_LINES_P
    speed_estimation_lines = settings.SPEED_ESTIMATION_LINES
    roads = settings.ROADS
    show_counts = settings.SHOW_COUNTS
    show_counts_P = settings.SHOW_COUNTS_P
    camera_height = settings.CAMERA_HEIGHT
    focal_length = settings.FOCAL_LENGTH
    pixel_length = settings.PIXEL_LENGTH
    resolution = settings.RESOLUTION
    vanishing_point = settings.VANISHING_POINT 
    
    frames_processed = 0
    frames_P_processed = 0
    processing_frame_rate = 0.0
    fps = []
    start_timer = time.time()
    veh_light=0
    ped_light=0
    
    
    vehicle_process = Processor(frame, detector, tracker, droi, show_droi, mcdf,
                                mctf, detection_interval, counting_lines, speed_estimation_lines,
                                show_counts, frames_processed, processing_frame_rate, roads,
                                veh_light, ped_light, camera_height, focal_length, pixel_length, resolution, vanishing_point)

    pedestrian_process = Processor(frame_P, detectorP, tracker, droi_P, show_droi_P, mcdf,
                                mctf, detection_interval, counting_lines_P, speed_estimation_lines,
                                show_counts_P, frames_P_processed, processing_frame_rate, roads,
                                veh_light, ped_light, camera_height, focal_length, pixel_length, resolution, vanishing_point)

    coordinateStored = CoordinateStore()
    record = settings.RECORD
    headless = settings.HEADLESS

    if record:
        # initialize video object to record counting
        output_video = cv2.VideoWriter(settings.OUTPUT_VIDEO_PATH, \
                                        cv2.VideoWriter_fourcc(*'MJPG'), \
                                        30, \
                                        (f_width, f_height))

    logger.info('Processing started.', extra={
        'meta': {
            'label': 'START_PROCESS',
            'counter_config': {
                'di': detection_interval,
                'mcdf': mcdf,
                'mctf': mctf,
                'detector': detector,
                'tracker': tracker,
                'use_droi': use_droi,
                'droi': droi,
                'show_droi': show_droi,
                'counting_lines': counting_lines
            },
        },
    })

    if not headless:
        # create two windows, one for vehicles and the other one for pedestrians.
        cv2.namedWindow('Vehicle')
        cv2.namedWindow('Pedestrian')
        veh_coordinate = coordinateStored
        ped_coordinate = coordinateStored
        cv2.setMouseCallback('Vehicle', veh_coordinate.mouse_callback, {'frame_width': f_width, 'frame_height': f_height})
        cv2.setMouseCallback('Pedestrian', ped_coordinate.mouse_callback, {'frame_width': f_P_width, 'frame_height': f_P_height})

    is_paused = False
    output_frame = None
    output_P_frame = None
    resized_frame = None
    resized_frame_P = None
    roi_setting_mode = False
    roi_setting_mode_P = False
    line_setting_mode = False
    line_setting_mode_P = False
    vpoint_setting_mode = False
    showUI = False
    extend_flag = False
    reduce_flag = False
    help_img_flag = False
    help_img_flag_time = 0
    help_img = cv2.imread('help.jpg')
    # main loop
    while is_cam or cap.get(cv2.CAP_PROP_POS_FRAMES) + 1 < cap.get(cv2.CAP_PROP_FRAME_COUNT) or cap_P.get(cv2.CAP_PROP_POS_FRAMES) + 1 < cap_P.get(cv2.CAP_PROP_FRAME_COUNT):
        clock_timer = time.time()
        k = cv2.waitKey(1) & 0xFF
        _len = len(veh_coordinate.points)
        _len_P = len(ped_coordinate.points)
        if k == ord('p'): # pause/play loop if 'p' key is pressed
            is_paused = False if is_paused else True
            logger.info('Loop paused/played.', extra={'meta': {'label': 'PAUSE_PLAY_LOOP', 'is_paused': is_paused}})

# =============================================================================
#       # need debug here         
#       if k == ord('s') and output_frame is not None: # save frame if 's' key is pressed
#             take_screenshot(output_frame)
# =============================================================================
            
        if k == ord('v'): # set roi if 'r' key is pressed
            roi_setting_mode = False if roi_setting_mode else True
            veh_coordinate.points = []
            _len = len(veh_coordinate.points)
            logger.info('Setting ROI for vehicle video.', extra={'meta': {'label': 'Setting ROI', 'ROI setting mode': roi_setting_mode}})         
        
        if k == ord('c'): # set roi if 'c' key is pressed
            roi_setting_mode_P = False if roi_setting_mode_P else True
            ped_coordinate.points = []
            _len_P = len(ped_coordinate.points)
            logger.info('Setting ROI for pedestrian video.', extra={'meta': {'label': 'Setting ROI', 'ROI setting mode': roi_setting_mode_P}})         
        if k == ord('q'): # end video loop if 'q' key is pressed
            logger.info('Loop stopped.', extra={'meta': {'label': 'STOP_LOOP'}})
            break
        
        if k == ord('z'):
            line_setting_mode = False if line_setting_mode else True
            veh_coordinate.points = []
            _len = len(veh_coordinate.points)
            logger.info('Setting line for vehicle video.', extra={'meta': {'label': 'Setting line', 'line_setting_mode': line_setting_mode}})

        if k == ord('x'):
            line_setting_mode_P = False if line_setting_mode_P else True
            ped_coordinate.points = []
            _len = len(ped_coordinate.points)
            logger.info('Setting line for pedestrian video.', extra={'meta': {'label': 'Setting line', 'line_setting_mode_P': line_setting_mode_P}})
            
        if k == ord('u'): # Show UI if 'u' key is pressed
            showUI = False if showUI else True
            logger.info('Showing UI.', extra={'meta': {'label': 'ShowUI'}})
        
        if k ==ord('a'):
            vpoint_setting_mode = False if vpoint_setting_mode  else True
            veh_coordinate.points = []
            _len = len(veh_coordinate.points)
            logger.info('Setting vanishing point for vehicle video.', extra={'meta': {'label': 'Setting vanishing point', 'vpoint_setting_mode': vpoint_setting_mode}})
        
        if k == ord('h'):
            help_img_flag = False if help_img_flag else True
        
        if k == ord('1'): # enter demo mode 1 if '1' key is pressed
            os.putenv('VIDEO',"./data/videos/few_vehicle_Trim.mp4")
            os.putenv('VIDEO_P',"./data/videos/personIsWaiting.mp4")
            os.putenv('DROI', '[(1072, 945), (969, 410), (1130, 380), (1907, 848)]')
            os.putenv('DROI_P','[(34, 565), (60, 988), (1911, 412), (1669, 243)]')
            os.putenv('COUNTING_LINES', "[{'label': 'A', 'line':[(1076, 940), (1898, 853)]}, ]")
            os.putenv('COUNTING_LINES_P', "[{'label': 'A', 'line':[(230, 567), (935, 812)]}, {'label': 'B', 'line':[(1336, 371), (1804, 416)]}]")
            os.putenv('VANISHING_POINT', '(1029, 351)')
            os.execl(sys.executable, os.path.abspath(__file__), *sys.argv) 
        
        if k == ord('2'): # enter demo mode 2 if '1' key is pressed
            os.putenv('VIDEO',"./data/videos/vah_normal.mp4")
            os.putenv('VIDEO_P',"./data/videos/few_people.mp4")
            os.putenv('DROI', '[(1072, 945), (969, 410), (1130, 380), (1907, 848)]')
            os.putenv('DROI_P','[(34, 565), (60, 988), (1911, 412), (1669, 243)]')
            os.putenv('COUNTING_LINES', "[{'label': 'A', 'line':[(1076, 940), (1898, 853)]}, ]")
            os.putenv('COUNTING_LINES_P', "[{'label': 'A', 'line':[(230, 567), (935, 812)]}, {'label': 'B', 'line':[(1336, 371), (1804, 416)]}]")
            os.putenv('VANISHING_POINT', '(1029, 351)')
            os.execl(sys.executable, os.path.abspath(__file__), *sys.argv) 
 
        if help_img_flag:
            cv2.imshow('Help',help_img)
            help_img_flag_time += 1
        if help_img_flag is False and help_img_flag_time > 0:
            cv2.destroyWindow('Help')
            
        if vpoint_setting_mode is True:
            roi_setting_mode = False
            roi_setting_mode_P = False 
            line_setting_mode = False
            line_setting_mode_P = False
            if _len >= 1:
              settings.replace_line('.env', 60,'VANISHING_POINT=' + str(veh_coordinate.points[0]) + '\n')
              os.putenv('VANISHING_POINT',str(veh_coordinate.points[0]))
              os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)  
        
        if roi_setting_mode is True:
            roi_setting_mode_P = False #prevent turning on 2 modes at the same time
            line_setting_mode = False
            line_setting_mode_P = False
            vpoint_setting_mode = False
            if _len >= 4:
                 settings.replace_line('.env', 4,'DROI=' + str(veh_coordinate.points[-4:]) + '\n')
                 os.putenv('DROI',str(veh_coordinate.points[-4:]))
                 os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)
            
        if roi_setting_mode_P is True:
            roi_setting_mode = False #prevent turning on 2 modes at the same time
            line_setting_mode = False
            line_setting_mode_P = False
            vpoint_setting_mode = False
            if _len_P >= 4:
                 os.putenv('DROI_P',str(ped_coordinate.points[-4:]))
                 settings.replace_line('.env', 5,'DROI_P=' + str(ped_coordinate.points[-4:]) + '\n') 
                 os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)
        
        if line_setting_mode is True:
            roi_setting_mode = False  #prevent turning on 2 modes at the same time
            roi_setting_mode_P = False 
            line_setting_mode_P = False
            vpoint_setting_mode = False
            if _len >= 2:
                 settings.replace_line('.env', 23, "COUNTING_LINES=[{'label': 'A', 'line':"+ str(veh_coordinate.points[-2:]) + "}, ]\n") 
                 os.putenv('COUNTING_LINES',"[{'label': 'A', 'line':"+ str(veh_coordinate.points[-2:]) + "}, ]")
                 os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)      
        
        if line_setting_mode_P is True:
            roi_setting_mode = False #prevent turning on 2 modes at the same time
            roi_setting_mode_P = False
            line_setting_mode = False
            vpoint_setting_mode = False 
            if _len >= 4:
                 settings.replace_line('.env', 24, "COUNTING_LINES_P=[{'label': 'A', 'line':"+ str(ped_coordinate.points[0:2]) + "}, {'label': 'B', 'line':"+ str(ped_coordinate.points[-2:])+"}]\n") 
                 os.putenv('COUNTING_LINES_P',"[{'label': 'A', 'line':"+ str(ped_coordinate.points[0:2]) + "}, {'label': 'B', 'line':"+ str(ped_coordinate.points[-2:])+"}]")
                 os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)           
        
        if is_paused is True:
            cv2.putText(frame, 'Pausing', (round(f_width/2),30),  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            time.sleep(0.5)
            continue

        _timer = cv2.getTickCount() # set timer to calculate processing frame rate
        
        extend_flag = vehicle_process.get_extend_flag() or pedestrian_process.get_extend_flag()
        reduce_flag = vehicle_process.get_reduce_flag() or pedestrian_process.get_reduce_flag()
        if extend_flag:
            start_timer = start_timer + 6
            extend_flag = False
            vehicle_process.clear_flag()
            pedestrian_process.clear_flag() 
            print('extended')
        if reduce_flag:
            start_timer = start_timer - 6
            reduce_flag = False
            vehicle_process.clear_flag()
            pedestrian_process.clear_flag() 
            print('reducing')
    
        veh_light, ped_light , light_timer = get_traffic_light_status(clock_timer-start_timer)
        if ret:
            vehicle_process.count(frame, frames_processed, processing_frame_rate)
            output_frame = vehicle_process.visualize(roi_setting_mode, is_paused, showUI, veh_light,ped_light,light_timer, roi_setting_mode_P,line_setting_mode,line_setting_mode_P,vpoint_setting_mode,help_img_flag)

            if record:
                output_video.write(output_frame)

            if not headless:
                debug_window_size = settings.DEBUG_WINDOW_SIZE
                resized_frame = cv2.resize(output_frame, debug_window_size)
                cv2.imshow('Vehicle', resized_frame)

        if ret_P:
            pedestrian_process.count(frame_P, frames_P_processed, processing_frame_rate)
            output_P_frame = pedestrian_process.visualize(roi_setting_mode, is_paused, showUI, veh_light ,ped_light,light_timer, roi_setting_mode_P,line_setting_mode,line_setting_mode_P,vpoint_setting_mode,help_img_flag)

            if record:
                output_video.write(output_P_frame)

            if not headless:
                debug_window_P_size = settings.DEBUG_WINDOW_P_SIZE
                resized_frame_P = cv2.resize(output_P_frame, debug_window_P_size)
                cv2.imshow('Pedestrian', resized_frame_P)
        
        processing_frame_rate = list_equalization(fps,round(cv2.getTickFrequency() / (cv2.getTickCount() - _timer), 10),30)
# =============================================================================
#         if len(fps) <= 30:
#             fps.append(round(cv2.getTickFrequency() / (cv2.getTickCount() - _timer), 10))
#         else:
#             fps.pop(0)
#             fps.append(round(cv2.getTickFrequency() / (cv2.getTickCount() - _timer), 10))
# =============================================================================
        frames_processed = round(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frames_count = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug('Frame processed.', extra={
            'meta': {
                'label': 'FRAME_PROCESS',
                'frames_processed': frames_processed,
                'frame_rate': processing_frame_rate,
                'frames_left': frames_count - frames_processed,
                'percentage_processed': round((frames_processed / frames_count) * 100, 2),
            },
        })
        ret, frame = cap.read()
        ret_P, frame_P = cap_P.read()

    # end capture, close window, close log file and video object if any
    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    if record:
        output_video.release()
    logger.info('Processing ended.', extra={'meta': {'label': 'END_PROCESS', 'counts': vehicle_process.counts}})


if __name__ == '__main__':
    run()

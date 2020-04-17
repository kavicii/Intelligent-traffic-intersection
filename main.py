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
from VehicleCounter import VehicleCounter
from util.debugger import CoordinateStore
init_logger()
logger = get_logger()


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

    detection_interval = settings.DI
    mcdf = settings.MCDF
    mctf = settings.MCTF
    detector = settings.DETECTOR
    tracker = settings.TRACKER
    # create detection region of interest polygon
    use_droi = settings.USE_DROI
    droi = settings.DROI \
            if use_droi \
            else [(0, 0), (f_width, 0), (f_width, f_height), (0, f_height)]
    show_droi = settings.SHOW_DROI
    counting_lines = settings.COUNTING_LINES
    speed_estimation_lines = settings.SPEED_ESTIMATION_LINES
    roads = settings.ROADS
    show_counts = settings.SHOW_COUNTS
    frames_processed = 0
    processing_frame_rate = 0.0
    fps = []
    
    
    vehicle_counter = VehicleCounter(frame, detector, tracker, droi, show_droi, mcdf,
                                     mctf, detection_interval, counting_lines, speed_estimation_lines , show_counts,frames_processed, processing_frame_rate, roads)
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
        # capture mouse events in the debug window
        cv2.namedWindow('Debug')
        cv2.setMouseCallback('Debug', coordinateStored.mouse_callback, {'frame_width': f_width, 'frame_height': f_height})

    is_paused = False
    output_frame = None
    resized_frame = None
    roi_setting_mode = False
    showUI = False
    # main loop
    while is_cam or cap.get(cv2.CAP_PROP_POS_FRAMES) + 1 < cap.get(cv2.CAP_PROP_FRAME_COUNT):
        k = cv2.waitKey(1) & 0xFF
        _len = len(coordinateStored.points)
        if k == ord('p'): # pause/play loop if 'p' key is pressed
            is_paused = False if is_paused else True
            logger.info('Loop paused/played.', extra={'meta': {'label': 'PAUSE_PLAY_LOOP', 'is_paused': is_paused}})

        if k == ord('s') and output_frame is not None: # save frame if 's' key is pressed
            take_screenshot(output_frame)
            
        if k == ord('r'): # set roi if 'r' key is pressed
            roi_setting_mode = False if roi_setting_mode else True
            logger.info('Setting ROI.', extra={'meta': {'label': 'Setting ROI', 'ROI setting mode': roi_setting_mode}})         
            
        if k == ord('q'): # end video loop if 'q' key is pressed
            logger.info('Loop stopped.', extra={'meta': {'label': 'STOP_LOOP'}})
            break

        if k == ord('u'): # Show UI if 'u' key is pressed
            showUI = False if showUI else True
            logger.info('Showing UI.', extra={'meta': {'label': 'ShowUI'}})
        
        if roi_setting_mode is True:
            # coordinateStored.__int__()
            if _len >= 4:
                 settings.replace_line('.env', 2,'DROI=' + str(coordinateStored.points[-4:]) + '\n') 
                 break
        
        if is_paused is True:
            cv2.putText(frame, 'Pausing', (round(f_width/2),30),  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            time.sleep(0.5)
            continue
        


        _timer = cv2.getTickCount() # set timer to calculate processing frame rate

        if ret:
            vehicle_counter.count(frame,frames_processed, processing_frame_rate)
            output_frame = vehicle_counter.visualize(roi_setting_mode,is_paused,showUI)

            if record:
                output_video.write(output_frame)

            if not headless:
                debug_window_size = settings.DEBUG_WINDOW_SIZE
                resized_frame = cv2.resize(output_frame, debug_window_size)
                cv2.imshow('Debug', resized_frame)

        if len(fps) <= 30:
            fps.append(round(cv2.getTickFrequency() / (cv2.getTickCount() - _timer), 10))
        else:
            fps.pop(0)
            fps.append(round(cv2.getTickFrequency() / (cv2.getTickCount() - _timer), 10))
        processing_frame_rate = sum(fps) / len(fps)
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

    # end capture, close window, close log file and video object if any
    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    if record:
        output_video.release()
    logger.info('Processing ended.', extra={'meta': {'label': 'END_PROCESS', 'counts': vehicle_counter.counts}})


if __name__ == '__main__':
    run()

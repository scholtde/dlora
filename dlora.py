# coding:utf-8

"""

"""


import logging
from logging.handlers import RotatingFileHandler
from logging.handlers import TimedRotatingFileHandler
import cv2
import numpy as np
import time
import sys
import ast
from termcolor import colored
from threading import Thread
from mvnc import mvncapi as mvnc
from armv7l.openvino.inference_engine import IENetwork, IEPlugin
# Custom packages
from packages.imutils.video import VideoStream
from packages.ai_stream import ai_stream as ai


class Dlora:
    def __init__(self, **kwargs):
        self.total_camera = 1
        self.streams = [0]
        self.cam_name = ["logitec c920 webcam"]
        self.cam_desc = []
        self.probability = []
        self.capture_list = []
        self.camera_list = []


    def configure(self):


    def run(self):
        # grab a list of all NCS devices plugged in to USB
        log =  "finding VPU devices..."
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
        # NCSDKv2 - Old API used to count devices, functionality not yet available for OpenVINO toolkit IE backend
        VPU_devices = mvnc.enumerate_devices()

        # if no devices found, exit the script
        if len(VPU_devices) == 0:
            log = "No VPU devices found"
            logging.info(log)
            print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)

        log = "found {} VPU devices".format(len(VPU_devices))
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
        vpu_device_count = len(VPU_devices)

        # Start threaded camera stream capture as well as initialise streamer for object detection
        for cam in range(self.total_camera):
            log = "starting video stream - " + self.cam_name[cam] + " : " + self.streams[cam]
            logging.info(log)
            print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)

            self.captures = VideoStream(self.streams[cam]).start()
            self.capture_list.append(self.captures)
            # Create ai object
            self.camera = ai.aiStreamer()
            # Initialize ai object members
            self.camera.capture = self.captures
            self.camera.cam_name = self.cam_name[cam]
            self.camera.cam_defined_objects = self.cam_defined_objects[cam]
            self.camera.AI_detection = self.object_detect_flags[cam]
            self.camera.probability = (int(self.probability[cam]))
            self.camera.cap_id = cam
            self.camera.vpu_schedule = self.vpu_s
            self.camera.zone_info_matrix = self.zones_info_matrix[cam]
            self.camera.zones_flag = self.zone_detect_flags[cam]

            # Setup the ai object
            self.camera.setup()

            self.camera_list.append(self.camera)

        # Display a wall of streams. This could be enhanced to grid form using placeholder images
        log = "starting video screen"
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)

        while True:
            try:
                # Process frames
                for i in range(len(self.camera_list)):
                    frame = self.camera_list[i].update()

                # Display opencv window of the captured frame
                cv2.namedWindow("CAM Capture", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("CAM Capture", 640, 480)
                cv2.imshow("CAM Capture", frame)

                key = cv2.waitKey(1) & 0xFF

                # if the `q` key is pressed, do cleanup and break from the loop
                if key == ord("q"):
                    log = "exiting..."
                    logging.info(log)
                    print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
                    self.end()

                    return False

            except KeyboardInterrupt:
                log = "exiting..."
                logging.info(log)
                print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
                self.end()

                return False

    def end(self):
        # Stop the stream list
        log = "stopping video streams"
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
        for i in range(len(self.capture_list)):
            self.capture_list[i].stop()

        cv2.destroyAllWindows()
        time.sleep(2)
        log = "application stopped"
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)


if __name__ == '__main__':
    log_file = "log/dlora.log"
    #handler = RotatingFileHandler(log_file, mode='a', maxBytes=20000000, backupCount=5)
    handler = TimedRotatingFileHandler(log_file,
                                       when='midnight',
                                       interval=1,
                                       backupCount=3,
                                       encoding=None,
                                       delay=False,
                                       utc=False)
    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO, handlers=[handler])

    # Start main
    while True:
        for i in range(101):  # for 0 to 100
            s = str(i) + '%'  # string for output
            print("[", colored("LOADING", 'red', attrs=['bold']), "]...", s, end='')
            print('\r', end='')  # use '\r' to go back
            time.sleep(0.01)  # sleep for 200ms
        print('\n')
        log = "OS platform: " + str(sys.platform)
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
        app = Dlora()
        app.configure()
        app.run()
        break

    sys.exit()

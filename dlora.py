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
from openvino.inference_engine import IENetwork, IEPlugin
# Custom packages
from packages.imutils.video import VideoStream
from packages.ai_stream import ai_stream as ai
# Bluetooth
from packages.blescan import blescan
import bluetooth._bluetooth as bluez

class Dlora:
    def __init__(self, **kwargs):
        # Camera and some model settings..
        # The below are intended to be defined externally by the user
        self.stream = "rtsp://192.168.1.101:554/av0_1"
        self.cam_name = "logitec c920 webcam"
        self.cam_desc = "just a normal webcam"
        self.cam_defined_objects = {"person": 30, "car": 50}
        self.object_detect_flag = True
        self.probability = 50  # if objects are not explicitly defined, this default value is used

        # grab a list of all NCS devices plugged in to USB
        log = "finding VPU devices..."
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
        log = "starting video stream - " + self.cam_name + " : " + str(self.stream)
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)

        # Setup a blank frame
        self.frame = np.zeros(shape=[360, 640, 3], dtype=np.uint8)

        # Start capturing frames from the camera stream
        self.capture = VideoStream(self.stream).start()

        # Create a AI streamer to pass frames to the object detector
        self.camera_ai = ai.aiStreamer()

        # Initialize ai object members
        self.camera_ai.capture = self.capture
        self.camera_ai.cam_name = self.cam_name
        self.camera_ai.cam_defined_objects = self.cam_defined_objects
        self.camera_ai.AI_detection = self.object_detect_flag
        self.camera_ai.probability = self.probability
        # Setup ble services
        self.ble_scanner, self.ble_sock = self.ble_services()
        self.camera_ai.ble_scanner = self.ble_scanner
        self.camera_ai.ble_sock = self.ble_sock
        # Setup the ai object
        self.camera_ai.setup()
        # Known UDID list
        self.known_things = [{"UDID": "0212233445566778899aabbccddeeff1",
                            "object_classification": "person",
                            "Details": "Dewald Scholtz"}]
        self.camera_ai.known_things = self.known_things
        if self.ble_scanner is not None:
            self.camera_ai.start_ble_loop()

    def ble_services(self):
        # BLE scanner
        device_err = False
        device_id = 0
        try:
            sock = bluez.hci_open_dev(device_id)
            log = "bluetooth device opened"
            logging.info(log)
            print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)

        except Exception as e:
            log = "accessing bluetooth device: " + str(e)
            logging.info(log)
            print("[", colored("ERROR", 'red', attrs=['bold']), "  ] " + log)
            device_err = True
            sock = None

        if device_err:
            blescanner = None
        else:
            blescanner = blescan.bleScan()
            blescanner.hci_le_set_scan_parameters(sock)
            blescanner.hci_enable_le_scan(sock)

        return blescanner, sock

    def run(self):
        # Display the stream
        log = "starting output video screen"
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)

        while True:
            try:
                # Process new frames
                self.frame = self.camera_ai.update()

                # Display opencv window of the captured frame
                cv2.namedWindow("CAM Capture", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("CAM Capture", 640, 480)
                cv2.imshow("CAM Capture", self.frame)

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
        log = "stopping video stream"
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
        self.capture.stop()
        self.camera_ai.stop_ble_loop()

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
        app.run()
        break

    sys.exit()

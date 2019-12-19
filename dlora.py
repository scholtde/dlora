# coding:utf-8

"""
// Copyright (c) 2019, RHIZOO CHRISTOS TECHNOLOGIES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of RHIZOO CHRISTOS TECHNOLOGIES nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
from multiprocessing import Process
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
        self.cam_name = "PTZ LAB IP Cam"
        self.cam_desc = "just a normal IP cam"
        self.cam_defined_objects = {"person": 50}
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
        # self.capture = VideoStream(self.stream).start()

        # Create a AI streamer to pass frames to the object detector
        # self.camera_ai = ai.aiStreamer()

        # Initialize ai object members
        self.camera_ai.capture = self.capture
        self.camera_ai.cam_name = self.cam_name
        self.camera_ai.cam_defined_objects = self.cam_defined_objects
        self.camera_ai.AI_detection = self.object_detect_flag
        self.camera_ai.probability = self.probability

        # Setup BLE services
        self.ble_scanner = None
        self.ble_sock = None
        self.ble_stop = False
        # self.camera_ai.ble_scanner = self.ble_scanner
        # self.camera_ai.ble_sock = self.ble_sock

        # Setup the ai object
        self.camera_ai.setup()

        # Known UDID list
        self.known_things = [{"UDID": "0212233445566778899aabbccddeeff1",
                            "object_classification": "person",
                            "Details": "Dewald Scholtz"}]
        self.camera_ai.ble_known_things = self.known_things

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
            blescanner = blescan.BleScan()
            blescanner.hci_le_set_scan_parameters(sock)
            blescanner.hci_enable_le_scan(sock)

        return blescanner, sock

    def start_ble_loop(self):
        t = Process(target=self.ble_loop, name="ble_loop", args=())
        t.daemon = True
        t.start()

    def ble_loop(self):
        while True:
            if self.ble_stop:
                return

            # self.ble_scanner_returned_device_dict = self.ble_scanner.parse_events(self.ble_sock, 1)
            ble_done = self.ble_scanner.parse_events(self.ble_sock, 1)

            # if ble_done:
            #     # Check if scanned dictionary buffer is empty
            #     if self.ble_scanner.discovered_devices_buffer:
            #         # Run through each device in the discovery buffer
            #         for device in self.ble_scanner.discovered_devices_buffer:
            #             # Run through each 'known' device
            #             for i in range(len(self.ble_known_things)):
            #                 # Check if the ID exist in the list of 'known' devices
            #                 if device["UDID"] in self.ble_known_things[i]["UDID"]:
            #                     # Check if the object 'class' exist for the specific AI model
            #                     if self.ble_known_things[i]["object_classification"] in self.dlora_class_vs_device:
            #                         # Add to buffer
            #                         if len(self.dlora_class_vs_device[self.ble_known_things[i]["object_classification"]]) != self.dlora_class_vs_device_buffer_length:
            #                             # Update the datails of the classification (dlora vs. discovered device)
            #                             self.dlora_class_vs_device[
            #                                 self.ble_known_things[i]["object_classification"]].append(
            #                                 self.ble_known_things[i]["Details"])
            #                         else:
            #                             self.dlora_class_vs_device[
            #                                 self.ble_known_things[i]["object_classification"]].pop(0)
            #                             # Update the datails of the classification (dlora vs. discovered device)
            #                             self.dlora_class_vs_device[
            #                                 self.ble_known_things[i]["object_classification"]].append(
            #                                 self.ble_known_things[i]["Details"])

    def stop_ble_loop(self):
        self.ble_stop = True

    def run(self):
        self.ble_scanner, self.ble_sock = self.ble_services()
        if self.ble_scanner is not None:
            self.ble_scanner.DEBUG = True
            self.start_ble_loop()

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
        self.stop_ble_loop()

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

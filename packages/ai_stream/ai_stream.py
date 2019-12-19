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
import time
import cv2
import numpy as np
from threading import Thread
from termcolor import colored

# Dlora
dlora_class_vs_device = {}

class aiStreamer:
    def __init__(self):
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] setting up AI VPU target devices")
        self.is_detect = False
        self.capture = None
        self.cam_name = None
        self.probability = None     # Used in default fallback
        self.AI_detection = None
        self.cam_defined_objects = None
        self.totals = []

        # BLE
        self.ble_scanner = None
        self.ble_sock = None
        self.ble_stop = False
        self.ble_known_things = None

        # Dlora
        self.dlora_class_vs_device_buffer_length = 30
        self.dlora_class_vs_device = {}

    def setup(self):
        log = "loading model.."
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
        self.net = cv2.dnn.readNetFromModelOptimizer("models/MobileNetSSD_deploy.xml", "models/MobileNetSSD_deploy.bin")
        self.lables_file = "models/labels/MobileNetSSD_labels.txt"
        self.colour_file = "models/labels/MobileNetSSD_colour"
        self.input_w = 300
        self.input_h = 300
        self.scale = 0.007843137
        self.mean = (127.5, 127.5, 127.5)
        self.swap_rb = 1
        model_name = "MobileNetSSD"

        # Notification Engine totals
        self.totals = []

        # Set the backend of the AI
        log = "setting up AI backends"
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

        # Setup CLASSES list
        log = "loading CLASS labels"
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
        with open(self.lables_file) as f:
            lines = f.read().splitlines()
        self.CLASSES = lines
        # Load all the objects defined in the model to serve as default fallback
        self.model_defined_objects = lines
        f.close()

        #  Iinit Dlora list
        # self.dlora_class_vs_device.clear()
        for d in range(len(self.model_defined_objects)):
            self.dlora_class_vs_device[self.model_defined_objects[d]] = []

        # Load colours
        with open(self.colour_file) as f:
            lines = f.read().splitlines()
            lines = list(lines)
        # read colour list from file, split lines and characters, convert to bgr, add to list
        lst = []
        for l in lines:
            string = str(l)
            line = string.strip("'()'")
            line = line.split(",")
            col = []
            b = int(line[2])
            col.append(b)
            g = int(line[1])
            col.append(g)
            r = int(line[0])
            col.append(r)
            lst.append(col)

        self.COLORS = lst
        f.close()

        log = "model: " + model_name + " loaded!"
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
        log = "AI for " + self.cam_name + " is ready!"
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)

        return

    def start_ble_loop(self):
        t = Thread(target=self.ble_loop, name="ble_loop", args=())
        t.daemon = True
        t.start()

    def stop_ble_loop(self):
        self.ble_stop = True

    def ble_loop(self):
        while True:
            if self.ble_stop:
                return

            # self.ble_scanner_returned_device_dict = self.ble_scanner.parse_events(self.ble_sock, 1)
            ble_done = self.ble_scanner.parse_events(self.ble_sock, 1)
            time.sleep(0.1)

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

    def update(self):
        # Read frame from the stream
        ret, self.frame = self.capture.read()

        # Resize the frame as desired
        w, h = 640, 360  # 16:9 aspect
        self.frame = cv2.resize(self.frame, (w, h))

        cam_label = self.cam_name

        # if a frame is not frozen, then...
        if ret is False:
            label = "Stream OK - Detecting..."
            self.frame = self.ai_object_detect()
            self.frame = cv2.putText(self.frame, label, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            self.frame = cv2.putText(self.frame, cam_label, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            return np.array(self.frame)

        # If there are issues with retrieving the frame
        else:
            label = "Stream ERROR! - Waiting..."

            # add the label in the frame
            self.frame = cv2.putText(self.frame, label, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            self.frame = cv2.putText(self.frame, cam_label, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            return np.array(self.frame)

    def ai_object_detect(self):
        obj_count = 0
        clean_frame = self.frame.copy()
        # Grab the frame dimensions and convert it to a blob
        (h, w) = self.frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (self.input_w, self.input_h)), self.scale,
                                     (self.input_w, self.input_h), self.mean)

        # Pass the blob through the DNN and gather the detections and predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        defined_objects = []

        # Extract objects from the dictionary which the user defined
        for object in self.cam_defined_objects:
            defined_objects.append(object)

        # If no objects are defined, then load all object classes from the model
        if not defined_objects:
            defined_objects = self.model_defined_objects

        for i in range(len(defined_objects)):
            self.totals.append(0)

        self.is_detect = False

        # Loop over the detections after the feed forward inference
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Check if there are tampering or false detection
            # (these constitute objects bigger than 90% of the frame in width or height)
            if (endX - startX)/w > 0.9 or (endY - startY)/h > 0.9:
                continue

            # Draw the prediction on the frame
            label = "{}: {:.2f}%".format(self.CLASSES[idx], confidence * 100)

            # This prevents the label from being cut off
            y = startY - 15 if startY - 15 > 15 else startY + 15

            # Only check defined objects
            if self.CLASSES[idx] in defined_objects:
                # Read the confidence value from the dictionary. If dictionary is empty, then use default fallback value
                if not self.cam_defined_objects:
                    defined_probability = self.probability/100
                else:
                    defined_probability = self.cam_defined_objects[self.CLASSES[idx]]/100

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence.
                if confidence > defined_probability:
                    obj_count += 1

                    found_object = self.CLASSES[idx]
                    dlora_label = "unknown"
                    if found_object in self.dlora_class_vs_device:
                        # Check if the list is not empty
                        if self.dlora_class_vs_device[found_object]:
                            # Remove the item from the buffer
                            dlora_label = self.dlora_class_vs_device[found_object].pop()

                    tp_x = int(startX + ((endX - startX) / 2))
                    tp_y = int(endY - ((endY - startY) / 2))

                    self.is_detect = True

                    # Draw Box
                    self.frame = cv2.rectangle(self.frame, (startX, startY), (endX, endY), self.COLORS[idx], 2)
                    cv2.putText(self.frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)
                    #cv2.line(self.frame, (tp_x-15, tp_y), (tp_x+15, tp_y), self.COLORS[idx], 3)
                    #cv2.line(self.frame, (tp_x, tp_y-15), (tp_x, tp_y+15), self.COLORS[idx], 3)
                    #cv2.circle(self.frame, (tp_x, tp_y), 10, (255, 255, 255), 2)

                    # Draw DLORA results
                    cv2.putText(self.frame, dlora_label, (startX + 130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                self.COLORS[idx], 2)

        return self.frame


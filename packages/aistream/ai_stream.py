# coding:utf-8

"""

"""

import logging
import cv2
import numpy as np
import ast
import time
import http.client as httplib
from termcolor import colored
from pathlib import Path
# Custom packages
import packages.imutils as iu


class aiStreamer:
    def __init__(self):
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] setting up AI VPU target devices")
        self.frame_count = 0
        self.frame_marker = 0
        self.frame_consecutive_detections = 0
        self.is_detect = False
        self.is_true_detect = False
        self.capture = None
        self.firstFrame = None
        self.last_time_motion = time.time()
        self.db_sqlite = None
        self.cam_name = None
        self.motion_detection = None
        self.ai_model = None
        self.vpu = None
        self.probability = None # Used in default fallback
        self.AI_detection = None
        self.AI_on_motion_detection = False
        self.json_post = None
        self.cap_id = None
        self.vpu_schedule = None
        self.zone_info_matrix = None
        self.zones_flag = None
        self.last_time = time.time()
        self.bot = None
        self.tamper = False
        self.tamper_count = 0
        self.cam_defined_objects = None
        self.home_dir = str(Path.home())

    def setup(self):
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] loading model")
        model_name = ""
        for retreive in self.db_sqlite.execute('select * from ai_models where model_id = ' + self.ai_model):
            self.net = cv2.dnn.readNetFromModelOptimizer(retreive[5], retreive[4])
            self.lables_file = retreive[3]
            self.colour_file = retreive[8]
            self.input_w = retreive[9]
            self.input_h = retreive[10]
            self.scale = retreive[11]
            self.mean = ast.literal_eval(retreive[12])
            self.swap_rb = retreive[13]
            model_name = retreive[1]

        # Notification Engine totals
        self.totals = []

        # Set the backend of the AI
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] setting up AI backends")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

        # Setup CLASSES list
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] loading CLASS labels")
        with open(self.lables_file) as f:
            lines = f.read().splitlines()
        self.CLASSES = lines
        # Load all the objects defined in the model to serve as default fallback
        self.model_defined_objects = lines
        f.close()

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

        self.writer_time = time.time()
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] AI for " + self.cam_name + " is ready!")

    def update(self):
        # Read frame from the stream
        ret, self.frame = self.capture.read()
        w, h = 640, 360  # 16:9 aspect
        self.frame = cv2.resize(self.frame, (w, h))

        # if a frame is not frozen, then...
        cam_label = self.cam_name
        if ret is False:
            label = "Stream OK - Detecting..."
            self.frame = self.ai_object_detect()
            self.frame = cv2.putText(self.frame, label, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            self.frame = cv2.putText(self.frame, cam_label, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Resize the frame
            # self.frame = iu.resize(self.frame, width=300)

            # cv2.imshow("CAM" + str(self.cap_id), new_frame)
            return np.array(self.frame)

        # If there are issues with retrieving the frame
        else:
            label = "Stream ERROR! - Waiting..."

            # add the label in the frame
            self.frame = cv2.putText(self.frame, label, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            self.frame = cv2.putText(self.frame, cam_label, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Resize the frame
            # self.frame = iu.resize(self.frame, width=300)

            return np.array(self.frame)

    def ai_object_detect(self):
        obj_count = 0
        clean_frame = self.frame.copy()
        # Grab the frame dimensions and convert it to a blob
        (h, w) = self.frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (self.input_w, self.input_h)), self.scale,
                                     (self.input_w, self.input_h), self.mean)

        # Pass the blob through the network and obtain the detections and predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        detect_array = []
        detected_objects = []
        defined_objects = []
        self.totals = []
        # Extract objects from the dictionary
        for object in self.cam_defined_objects:
            defined_objects.append(object)
        # If no objects are defined, then load all the model objects
        if not defined_objects:
            defined_objects = self.model_defined_objects
        for i in range(len(defined_objects)):
            self.totals.append(0)
        self.bot.defined_objects[self.cap_id] = defined_objects

        self.is_detect = False
        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]
            point = None


            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            zone_id = 0
            # Check if there are tampering or false detection
            if (endX - startX)/w > 0.9 or (endY - startY)/h > 0.9:
                continue

            # Draw the prediction on the frame
            label = "{}: {:.2f}%".format(self.CLASSES[idx], confidence * 100)

            # Don't cut-off label
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
                    """if self.CLASSES[idx] != object:
                      # Skip rest of the statements if the object is not defined
                      continue
                    """
                    # Test point for Zone detection (object in zone or not?)
                    tp_x = int(startX + ((endX - startX) / 2))
                    tp_y = int(endY - ((endY - startY) / 2))
                    if self.CLASSES[idx] == "person":
                        tp_y = int(endY - ((endY - startY) / 5)) # move offset 1/5th from bounding box bottom
                    if self.CLASSES[idx] == "car":
                        tp_y = int(endY - ((endY - startY) / 3)) # move offset 1/3th from bounding box bottom

                    self.is_detect = True

                    # Draw Box
                    self.frame = cv2.rectangle(self.frame, (startX, startY), (endX, endY), self.COLORS[idx], 2)
                    # Check if user busy annotating or not and if it is ready for next frame and for which cam
                    if self.bot.annotation_mode and \
                       self.bot.annotation_ready is False and \
                       self.bot.annotation_ready_next and \
                       self.bot.annotation_selected_cam == self.cap_id:
                        cv2.putText(self.frame, str(obj_count), (tp_x, tp_y), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                    self.COLORS[idx], 2)
                        self.bot.annotation_boxes.append([obj_count, startX, startY, endX, endY])
                    else:
                        cv2.putText(self.frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)
                        cv2.line(self.frame, (tp_x-15, tp_y), (tp_x+15, tp_y), self.COLORS[idx], 3)
                        cv2.line(self.frame, (tp_x, tp_y-15), (tp_x, tp_y+15), self.COLORS[idx], 3)
                        cv2.circle(self.frame, (tp_x, tp_y), 10, (255, 255, 255), 2)

                    # check if object is in a zone
                    if self.zones_flag is True:
                        inzone = False
                        for i in range(len(self.zone_info_matrix[3])):
                            zone_id = i + 1
                            zone_test = cv2.pointPolygonTest(self.zone_info_matrix[3][i], (tp_x, tp_y), False)

                            if zone_test != -1:
                                inzone = True
                                # Prepare array of all the objects detected
                                self.totals[defined_objects.index(found_object)] += 1
                                detected_objects.append(
                                    dict(zone_id=str(zone_id),
                                         classification=found_object,
                                         probability=str(round(confidence * 100, 2))))

                                detect_array.append(
                                    dict(capture_id=str(self.cap_id),
                                         zone_id=str(zone_id),
                                         classification=found_object,
                                         probability=str(round(confidence * 100, 2)),
                                         bb_coordinates=[str(startX), str(startY), str(endX), str(endY)]))
                        # there were objects, but they were not in a zone
                        self.is_detect = inzone
                    else:
                        self.totals[defined_objects.index(found_object)] += 1
                        detected_objects.append(
                            dict(zone_id=str(zone_id),
                                 classification=found_object,
                                 probability=str(round(confidence * 100, 2))))

                        detect_array.append(
                            dict(capture_id=str(self.cap_id),
                                 zone_id=str(zone_id),
                                 classification=found_object,
                                 probability=str(round(confidence * 100, 2)),
                                 bb_coordinates=[str(startX), str(startY), str(endX), str(endY)]))

        # Accumulate the detections
        self.accumulate(detect_array, self.is_detect)
        # Were there any ai detections
        if self.is_detect:
            # Is the positive detection a consecutive frame from previous detections
            if self.frame_count - 1 == self.frame_marker:
                # Increase consecutive frame positives
                self.frame_consecutive_detections += 1
            else:
                # Set marker with detection positive frame count for a next time test
                self.frame_marker = self.frame_count

            # If there are more consecutive positive frames than defined, set the ai detections as "True Positives"
            if self.frame_consecutive_detections > 2:
                # Reset consecutive positive detections
                self.frame_consecutive_detections = 0
                # Let the bot know there are detections
                self.bot.ai_detections_list[self.cap_id] = self.cam_name + \
                                                           "\nObjects: " + str(detected_objects) + \
                                                           "\n\nTotal Objects: " + str(sum(self.totals)) + \
                                                           "\nOccurred at: " + str(time.ctime())
                self.bot.ai_detections_list_notifications[self.cap_id] = self.totals.copy()
                self.is_true_detect = True

                #if time.time() - self.writer_time > 1:
                #    self.writer_time = time.time()
                if self.bot.annotation_mode and \
                        self.bot.annotation_ready is False and \
                        self.bot.annotation_ready_next and \
                        self.bot.annotation_selected_cam == self.cap_id:
                    self.bot.annotation_frame = self.frame
                    self.bot.annotation_frame_shape = (w, h)
                    self.bot.annotation_frame_save = clean_frame
                    self.bot.annotation_ready = True

        else:
            self.bot.ai_detections_list_notifications[self.cap_id] = None

        # Reset totals
        for i in range(len(self.totals)):
            self.totals[i] = 0

        return self.frame


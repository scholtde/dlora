# coding:utf-8

"""

"""

import logging
import cv2
import numpy as np
from threading import Thread
from termcolor import colored


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
        self.known_things = None

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
            returnedDict = self.ble_scanner.parse_events(self.ble_sock, 1)
            #print(returnedDict)
            if returnedDict["UDID"] in self.known_things["UDID"]:
                print("known person")

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

        # Pass the blob through the DNN and gather the detections and predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        detect_array = []
        detected_objects = []
        defined_objects = []

        # Extract objects from the dictionary which the user defined
        for object in self.cam_defined_objects:
            defined_objects.append(object)

        # If no objects are defined, then load all the model objects
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
                    # if self.CLASSES[idx] == "person":
                    #     tp_y = int(endY - ((endY - startY) / 5)) # move offset 1/5th from bounding box bottom
                    # if self.CLASSES[idx] == "car":
                    #     tp_y = int(endY - ((endY - startY) / 3)) # move offset 1/3th from bounding box bottom

                    self.is_detect = True

                    # Draw Box
                    self.frame = cv2.rectangle(self.frame, (startX, startY), (endX, endY), self.COLORS[idx], 2)
                    cv2.putText(self.frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)
                    cv2.line(self.frame, (tp_x-15, tp_y), (tp_x+15, tp_y), self.COLORS[idx], 3)
                    cv2.line(self.frame, (tp_x, tp_y-15), (tp_x, tp_y+15), self.COLORS[idx], 3)
                    cv2.circle(self.frame, (tp_x, tp_y), 10, (255, 255, 255), 2)

                    # self.totals[defined_objects.index(found_object)] += 1
                    # detected_objects.append(
                    #     dict(zone_id=str(zone_id),
                    #          classification=found_object,
                    #          probability=str(round(confidence * 100, 2))))
                    #
                    # detect_array.append(
                    #     dict(capture_id=str(self.cap_id),
                    #          zone_id=str(zone_id),
                    #          classification=found_object,
                    #          probability=str(round(confidence * 100, 2)),
                    #          bb_coordinates=[str(startX), str(startY), str(endX), str(endY)]))

        # # Accumulate the detections
        # self.accumulate(detect_array, self.is_detect)
        # # Were there any ai detections
        # if self.is_detect:
        #     # Is the positive detection a consecutive frame from previous detections
        #     if self.frame_count - 1 == self.frame_marker:
        #         # Increase consecutive frame positives
        #         self.frame_consecutive_detections += 1
        #     else:
        #         # Set marker with detection positive frame count for a next time test
        #         self.frame_marker = self.frame_count
        #
        #     # If there are more consecutive positive frames than defined, set the ai detections as "True Positives"
        #     if self.frame_consecutive_detections > 2:
        #         # Reset consecutive positive detections
        #         self.frame_consecutive_detections = 0
        #         # Let the bot know there are detections
        #         self.bot.ai_detections_list[self.cap_id] = self.cam_name + \
        #                                                    "\nObjects: " + str(detected_objects) + \
        #                                                    "\n\nTotal Objects: " + str(sum(self.totals)) + \
        #                                                    "\nOccurred at: " + str(time.ctime())
        #         self.bot.ai_detections_list_notifications[self.cap_id] = self.totals.copy()
        #         self.is_true_detect = True
        #
        #         #if time.time() - self.writer_time > 1:
        #         #    self.writer_time = time.time()
        #         if self.bot.annotation_mode and \
        #                 self.bot.annotation_ready is False and \
        #                 self.bot.annotation_ready_next and \
        #                 self.bot.annotation_selected_cam == self.cap_id:
        #             self.bot.annotation_frame = self.frame
        #             self.bot.annotation_frame_shape = (w, h)
        #             self.bot.annotation_frame_save = clean_frame
        #             self.bot.annotation_ready = True
        #
        # else:
        #     self.bot.ai_detections_list_notifications[self.cap_id] = None
        #
        # # Reset totals
        # for i in range(len(self.totals)):
        #     self.totals[i] = 0

        return self.frame


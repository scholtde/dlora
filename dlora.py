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
from packages.jsonpost import jsonpost as jp
from packages.vpu_scheduler import vpu_scheduler as vs
from packages.bot import bot as cb
from packages.db_engines import db_engines
from packages.db_watch import db_watch


class ET:
    def __init__(self, iter, cbot, **kwargs):
        log = "startup " + str(iter)
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
        self.db_sqlite = db_engines.SQLite()
        self.cam_matrix = []
        self.zone_matrix = []
        self.schedule_matrix = []
        self.matrix_check_busy = False
        self.total_camera = 0
        self.limit = 0
        self.capture_list = []
        self.camera_list = []
        self.streams = []
        self.cam_name = []
        self.cam_desc = []
        self.object_detect_flags = []
        self.probability = []
        self.zone_detect_flags = []
        self.motion_detect_flags = []
        self.cam_defined_objects = []
        self.cam_gps_location = []
        self.cam_facing_direction = []
        self.zones_info_matrix = []
        self.zone_list = []
        self.zone_colour_list = []
        self.restart_signal = False
        self.bot = cbot

    def configure(self):
        for matrix in self.db_sqlite.execute('select stream_URL, stream_description, camera_name, object_detection, od_confidence, zone_detection, motion_detect, cam_defined_objects, cam_gps_location, cam_facing_direction, model_id, vpu_id from cams'):
            self.cam_matrix.append(matrix)

        for matrix in self.db_sqlite.execute('SELECT * FROM zones'):
            self.zone_matrix.append(matrix)

        for cams in self.db_sqlite.execute('SELECT COUNT(cams.object_detection) FROM cams'):
            self.total_camera = int(cams[0])

        for cams in self.db_sqlite.execute('select device_max_streams from device_config where device_id = 1'):
            self.limit = int(cams[0])

        # Ensure total camera streams does not exceed device design limits
        if self.total_camera > self.limit:
            self.total_camera = self.limit

        # Setup camera streams located in the DB
        for cams in self.db_sqlite.execute('SELECT * FROM cams limit ' + str(self.limit)):
            self.streams.append(cams[1])
            self.cam_desc.append(cams[2])
            self.cam_name.append(cams[3])
            self.object_detect_flags.append(ast.literal_eval(cams[4]))
            self.probability.append(cams[5])
            self.zone_detect_flags.append(ast.literal_eval(cams[6]))
            self.motion_detect_flags.append(ast.literal_eval(cams[7]))
            self.schedule_matrix.append(cams[8])
            # Convert defined objects JSON field to dictionary
            self.cam_defined_objects.append(dict(ast.literal_eval(cams[9])))
            self.cam_gps_location.append(cams[10])
            self.cam_facing_direction.append(cams[11])

        # Assemble the zone info matrix
        zones_id = []
        zones_name = []
        zones_description = []
        zones_poly = []
        zones_colour = []
        poly_center = []
        for i in range(self.total_camera):
            stream_id = i + 1
            for zone in self.db_sqlite.execute(
                    "select zone_id, zone_name, zone_description, zone_poly, zone_colour from zones left join cams on zones.stream_id = cams.stream_id where cams.stream_id = CAST(" + str(
                      stream_id) + " AS INTEGER)"):
                zones_id.append(zone[0])
                zones_name.append(zone[1])
                zones_description.append(zone[2])
                # Strip away the string characters and check string and convert the field to numpy array
                ret_zones = str(zone[3])
                ret_zones = ret_zones.strip("'(,)'")
                ret_zones = ast.literal_eval(ret_zones)
                zones_poly.append(np.array([ret_zones], np.int32))
                zones_colour.append(zone[4])
                poly_center.append(self.centroid(ret_zones))
            self.zone_list.append(zones_poly)
            self.zone_colour_list.append(zones_colour)
            zones_info_matrix = [zones_id, zones_name, zones_description, zones_poly, zones_colour, poly_center]
            self.zones_info_matrix.append(zones_info_matrix)
            # Empty for the next camera
            zones_id = []
            zones_name = []
            zones_description = []
            zones_poly = []
            zones_colour = []
            poly_center = []

        # Create API
        self.post = jp.post_packet(self.total_camera)
        self.post.start()

        # Start a VPU sheduler
        self.vpu_s = vs.scheduler(self.total_camera, 0.1)
        self.vpu_s.start()

        # Create watchdog
        self.wd = db_watch.DBWatch()
        self.wd.cam_matrix = self.cam_matrix
        self.wd.zone_matrix = self.zone_matrix
        self.wd.schedule_matrix = self.schedule_matrix
        self.wd.start()

        # Init chat bot arrays
        self.bot.reset()
        self.bot.cam_settings = []
        self.bot.ai_detections_list = []
        self.bot.cam_ip = []
        self.bot.ai_model = "1"
        for i in range(self.total_camera):
            self.bot.cam_settings.append("CAM" + str(i+1) +
                                         "\nStream URL: " + self.streams[i] +
                                         "\nStream Desc: " + self.cam_desc[i] +
                                         "\nObject Detection: " + str(self.object_detect_flags[i]) +
                                         "\nDefault Confidence: " + str(self.probability[i]) +
                                         "\nZone Detection: " + str(self.zone_detect_flags[i]) +
                                         "\nDetect on Motion: " + str(self.motion_detect_flags[i]) +
                                         "\nDefined Objects: " + str(self.cam_defined_objects[i]) +
                                         "\nGPS Location: " + self.cam_gps_location[i] +
                                         "\nFacing Direction: " + self.cam_facing_direction[i])

            self.bot.ai_detections_list.append("CAM" + str(i+1) + " - nothing found yet")
            self.bot.ai_detections_list_notifications.append(None)
            self.bot.cam_ip.append(self.streams[i])
        self.bot.cam_gps_location = self.cam_gps_location
        self.bot.start()

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

        # Init AI model scheduler
        """
        device_ids = [int(i) for i in range(vpu_device_count)]
        model_xml = "models/MobileNetSSD_deploy.xml"
        x = classifier_Scheduler(device_ids, model_xml)
        """

        # Start threaded camera stream capture as well as initialise streamer for object detection
        for cam in range(self.total_camera):
            log = "starting video stream - " + self.cam_name[cam] + " : " + self.streams[cam]
            logging.info(log)
            print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
            print("[", colored("SETUP", 'blue', attrs=['bold']),
                  "  ] settings: object detection = " + str(self.object_detect_flags[cam]) + ", zone detection = " + str(
                    self.zone_detect_flags[cam]) + ", motion detection = " + str(self.motion_detect_flags[cam]))

            self.captures = VideoStream(self.streams[cam]).start()
            self.capture_list.append(self.captures)
            # Create ai object
            self.camera = ai.aiStreamer()
            # Initialize ai object members
            self.camera.capture = self.captures
            self.camera.db_sqlite = self.db_sqlite
            self.camera.cam_name = self.cam_name[cam]
            self.camera.motion_detection = self.motion_detect_flags[cam]
            self.camera.ai_model = "1"
            self.camera.vpu = "1"
            self.camera.cam_defined_objects = self.cam_defined_objects[cam]
            self.camera.AI_detection = self.object_detect_flags[cam]
            self.camera.probability = (int(self.probability[cam]))
            self.camera.json_post = self.post
            self.camera.cap_id = cam
            self.camera.vpu_schedule = self.vpu_s
            self.camera.zone_info_matrix = self.zones_info_matrix[cam]
            self.camera.zones_flag = self.zone_detect_flags[cam]
            self.camera.bot = self.bot
            # Setup the ai object
            self.camera.setup()

            self.camera_list.append(self.camera)

        # Display a wall of streams. This could be enhanced to grid form using placeholder images
        log = "starting video wall"
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)

        # do once to fill the list
        wall = []
        for i in range(len(self.camera_list)):
            # Create blank images to fill the array
            frame = np.zeros(shape=[360, 640, 3], dtype=np.uint8)
            self.bot.latest_ai_frame.append(frame)
            self.bot.latest_ai_bb_frame.append(frame)

            if i == 0:
                wall = frame
            if i > 0:
                #wall = np.concatenate((wall, frame), axis=1)
                wall = np.vstack([wall, frame])

        if len(wall) > 0:
            self.bot.latest_ai_wall = wall

        while True:
            try:
                # Process frames
                for i in range(len(self.camera_list)):
                    frame = self.camera_list[i].update()

                    # update chat bot frames
                    #if self.bot.pic_request:
                    #    self.bot.latest_ai_frame[i] = frame
                    #self.bot.latest_ai_frame[i] = frame

                    if i == 0:
                        wall = frame
                    if i > 0:
                        #wall = np.concatenate((wall, frame), axis=1)
                        wall = np.vstack([wall, frame])

                # Display a walled opencv window
                # cv2.namedWindow("CAM Wall", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("CAM Wall", 640, 480)
                # cv2.imshow("CAM Wall", wall)

                # Check if the chat bot wants something
                #if self.bot.pic_request:
                self.bot.latest_ai_wall = wall
                self.bot.cam_settings_request_ready = True
                self.bot.detections_request_ready = True

                # Check db changes, if changes then initiate restart
                if self.matrix_check_busy is False:
                    if self.changes():
                        self.end()

                        return True

                key = cv2.waitKey(1) & 0xFF

                # if the `q` key is pressed, do cleanup and break from the loop
                if key == ord("q"):
                    log = "exiting..."
                    logging.info(log)
                    print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
                    self.end()

                    return False

                if key == ord("r"):
                    self.end()

                    return True

            except KeyboardInterrupt:
                log = "exiting..."
                logging.info(log)
                print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
                self.end()

                return False

    def changes(self):
        self.matrix_check_busy = True
        t = Thread(target=self.check)
        t.daemon = True
        t.start()
        self.matrix_check_busy = False
        return self.restart_signal

    def check(self):
        if self.wd.cam_matrix != self.cam_matrix:
            for row in range(len(self.wd.cam_matrix)):
                for col in range(len(self.wd.cam_matrix[row])):
                    if self.cam_matrix[row][col] != self.wd.cam_matrix[row][col]:
                        # Col=0 contains stream URL, only restart device when this changes
                        if col == 0:
                            self.wd.cam_matrix = []
                            self.cam_matrix = []
                            self.restart_signal = True

                            return

            self.cam_matrix = self.wd.cam_matrix

            db = db_engines.SQLite()
            self.cam_desc = []
            self.object_detect_flags = []
            self.probability = []
            self.zone_detect_flags = []
            self.motion_detect_flags = []
            self.cam_defined_objects = []
            self.cam_gps_location = []
            self.cam_facing_direction = []

            # Setup camera streams located in the DB
            for cams in db.execute('SELECT * FROM cams limit ' + str(self.limit)):
                self.cam_desc.append(cams[2])
                self.object_detect_flags.append(ast.literal_eval(cams[4]))
                self.probability.append(cams[5])
                self.zone_detect_flags.append(ast.literal_eval(cams[6]))
                self.motion_detect_flags.append(ast.literal_eval(cams[7]))
                # Convert defined objects JSON field to dictionary
                self.cam_defined_objects.append(dict(ast.literal_eval(cams[9])))
                self.cam_gps_location.append(cams[10])
                self.cam_facing_direction.append(cams[11])

            # re-initialise streamer for object detection
            for cam in range(len(self.camera_list)):
                log = "updating video stream - " + self.cam_name[cam]
                logging.info(log)
                print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
                print("[", colored("SETUP", 'blue', attrs=['bold']),
                      "  ] settings: object detection = " + str(
                          self.object_detect_flags[cam]) + ", zone detection = " + str(
                          self.zone_detect_flags[cam]) + ", motion detection = " + str(
                          self.motion_detect_flags[cam]))

                # Initialize ai object members
                self.camera_list[cam].motion_detection = self.motion_detect_flags[cam]
                self.camera_list[cam].cam_defined_objects = self.cam_defined_objects[cam]
                self.camera_list[cam].AI_detection = self.object_detect_flags[cam]
                self.camera_list[cam].probability = (int(self.probability[cam]))
                self.camera_list[cam].zones_flag = self.zone_detect_flags[cam]

            # Update bot
            self.bot.cam_settings = []
            for i in range(len(self.camera_list)):
                self.bot.cam_settings.append("CAM" + str(i + 1) +
                                             "\nStream URL: " + self.streams[i] +
                                             "\nStream Desc: " + self.cam_desc[i] +
                                             "\nObject Detection: " + str(self.object_detect_flags[i]) +
                                             "\nDefault Confidence: " + str(self.probability[i]) +
                                             "\nZone Detection: " + str(self.zone_detect_flags[i]) +
                                             "\nDetect on Motion: " + str(self.motion_detect_flags[i]) +
                                             "\nDefined Objects: " + str(self.cam_defined_objects[i]) +
                                             "\nGPS Location: " + self.cam_gps_location[i] +
                                             "\nFacing Direction: " + self.cam_facing_direction[i])
            self.bot.cam_gps_location = self.cam_gps_location

            return

        if self.wd.zone_matrix != self.zone_matrix:
            # Assemble the zone info matrix
            db = db_engines.SQLite()
            self.zones_info_matrix = []
            zones_id = []
            zones_name = []
            zones_description = []
            zones_poly = []
            zones_colour = []
            poly_center = []
            for i in range(len(self.camera_list)):
                stream_id = i + 1
                for zone in db.execute(
                        "select zone_id, zone_name, zone_description, zone_poly, zone_colour from zones left join cams on zones.stream_id = cams.stream_id where cams.stream_id = CAST(" + str(
                            stream_id) + " AS INTEGER)"):
                    zones_id.append(zone[0])
                    zones_name.append(zone[1])
                    zones_description.append(zone[2])
                    # Strip away the string characters and check string and convert the field to numpy array
                    ret_zones = str(zone[3])
                    ret_zones = ret_zones.strip("'(,)'")
                    ret_zones = ast.literal_eval(ret_zones)
                    zones_poly.append(np.array([ret_zones], np.int32))
                    zones_colour.append(zone[4])
                    poly_center.append(self.centroid(ret_zones))
                self.zone_list.append(zones_poly)
                self.zone_colour_list.append(zones_colour)
                zones_info_matrix = [zones_id, zones_name, zones_description, zones_poly, zones_colour, poly_center]
                self.zones_info_matrix.append(zones_info_matrix)
                # Empty for the next camera
                zones_id = []
                zones_name = []
                zones_description = []
                zones_poly = []
                zones_colour = []
                poly_center = []

            # re-initialise streamer for object detection
            for cam in range(len(self.camera_list)):
                self.camera_list[cam].zone_info_matrix = self.zones_info_matrix[cam]

            self.zone_matrix = self.wd.zone_matrix
            #self.restart_signal = True
            return

        if self.wd.schedule_matrix != self.schedule_matrix:
            self.schedule_matrix = self.wd.schedule_matrix
            self.bot.new_schedule = True
            return

    def end(self):
        # Stop notification engine
        self.bot.stop()

        # Stop the VPU scheduler
        log = "stopping VPU scheduler"
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
        self.vpu_s.stop()
        del self.vpu_s

        # Stop db watch
        self.wd.stop()
        del self.wd

        # Stop the stream list
        log = "stopping video streams"
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
        for i in range(len(self.capture_list)):
            self.capture_list[i].stop()

        cv2.destroyAllWindows()
        time.sleep(2)
        # Close sockets
        log = "closing sockets"
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)
        time.sleep(5)
        self.post.stop()
        log = "application stopped"
        logging.info(log)
        print("[", colored("INFO", 'green', attrs=['bold']), "   ] " + log)

    def display(self, frame, id):
        cv2.imshow("CAM" + str(id), frame)

    def centroid(self, poly):
      x_list = [point[0] for point in poly]
      y_list = [point[1] for point in poly]
      length = len(poly)
      x = int(sum(x_list) / length)
      y = int(sum(y_list) / length)

      return (x, y)

if __name__ == '__main__':
    log_file = "../et.log"
    #handler = RotatingFileHandler(log_file, mode='a', maxBytes=20000000, backupCount=5)
    handler = TimedRotatingFileHandler(log_file,
                                       when='midnight',
                                       interval=1,
                                       backupCount=3,
                                       encoding=None,
                                       delay=False,
                                       utc=False)
    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO, handlers=[handler])

    #logging.basicConfig(filename='../et.log', filemode='a', format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    # Connect to db first
    db = db_engines.SQLite()

    # How many restarts?
    iteration = 1

    # Create chat bot
    key = None
    commander = None
    for bots in db.execute("SELECT * FROM bots WHERE bot_id='1'"):
        key = bots[2]
        if key == "not_defined" or len(key) == 0:
            key = "123456789:BBFTbIs8P6EbFZDN0tI8NR5jzGV4RiyieI8"

    for user in db.execute('SELECT user_id, role FROM bot_users'):
        if user[1] == "commander":
            commander = user[0]
            if commander == "0":
                f = open("settings/nuc", 'r')
                if f.mode == 'r':
                    commander = int(f.read()) / 379
                    bot = cb.ChatBot(key, int(commander))
                    bot.nuc = True
                    bot.welcome()
                    bot.setup()
            else:
                bot = cb.ChatBot(key, int(commander))
                bot.nuc = False
                bot.welcome()
                bot.setup()

    del db

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
        app = ET(iter=iteration, cbot=bot)
        app.configure()
        restart = app.run()
        if restart:
            del app
            for i in range(101):  # for 0 to 100
                s = str(i) + '%'  # string for output
                log = "restarting application..."
                logging.warning(log)
                print("[", colored("WARNING", 'yellow', attrs=['bold']),"] restarting application...",s, end='')
                print('\r', end='')  # use '\r' to go back
                time.sleep(0.01)
            print('\n')
            iteration += 1
        else:
            break
    bot.stop()

    sys.exit()

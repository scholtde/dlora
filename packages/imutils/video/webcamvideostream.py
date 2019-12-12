# import the necessary packages
from threading import Thread
import cv2
import time

class WebcamVideoStream:
	def __init__(self, src=0, name="WebcamVideoStream"):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.placeholder = cv2.imread("img/ph.jpg")
		self.frozen = False
		self.source = src
		self.grabbed = False
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
		self.notOpened = False
		# initialize the thread name
		self.name = name
		self.grab()
		self.last_time = time.time()


	def grab(self):
		self.source = "rtspsrc location=" + self.source + " ! application/x-rtp, media=video, clock-rate=90000, encoding-name=H264 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink sync=false"
		self.stream = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
		if (self.stream.isOpened()) == True:
			self.notOpened = False
			(self.grabbed, frame) = self.stream.read()
			if self.grabbed == False:
				self.notOpened = True
				self.stream.release()
				self.frame = self.placeholder
				self.frozen = True
			else:
				self.frame = frame
		else:
			self.grabbed = False
			#self.stopped = True
			self.notOpened = True
			self.stream.release()
			self.frozen = True
			self.frame = self.placeholder

	def start(self):
		if self.notOpened == False:
			# start the thread to read frames from the video stream
			t = Thread(target=self.update, name=self.name, args=())
			t.daemon = True
			t.start()
			return self
		else:
			return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				self.stream.release()
				return

			# otherwise, read the next frame from the stream
			#if self.grabbed == True:
			(self.grabbed, frame) = self.stream.read()
			#time.sleep(0.05) # Slows down stream to process test videos

			if self.grabbed == False:
				self.frame = self.placeholder
				self.frozen = True
				try:
					try:
						self.stream.release()
						del self.stream
					except:
						pass
					time.sleep(60)
					print("")
					print("Retrying stream - " + str(self.source))
					print("")
					self.grab()
				except:
					print("")
					print("Stream Failed! - " + str(self.source))
					print("")
			else:
				self.frame = frame
				self.frozen = False

	def read(self):
		if self.notOpened == False:
			if self.frozen:
				self.frame = self.placeholder
				# return the frame most recently read
				return self.frozen, self.frame
			else:
				# return the frame most recently read
				return self.frozen, self.frame
		else:
			#print("[ ERROR ] video stream or file opening failed")
			self.frame = self.placeholder
			return self.frozen, self.frame


	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

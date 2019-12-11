import blescan
import sys
import datetime
import time
import os

import bluetooth._bluetooth as bluez
import threading
import thread



dev_id = 0    
scanning = True
debug = True

	
#Scanner function
def scan():
	try:
		sock = bluez.hci_open_dev(dev_id)

	except Exception as e:
		print("ERROR: Accessing bluetooth device: " + str(e))
		sys.exit(1)

	blescan.hci_le_set_scan_parameters(sock)
	blescan.hci_enable_le_scan(sock)
	
	#Keep scanning until the manager is told to stop.
	while scanning:
		
		returnedList = blescan.parse_events(sock, 10)
		
		for beacon in returnedList:
			beaconParts = beacon.split(",")
			
			if (debug):
					print(" Tilt Device Found (UUID " + beaconParts[1] + "): " + str(beaconParts))
			

#Stop Scanning function
def stop():
	scanning = False


scan()

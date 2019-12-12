import blescan
import sys
import bluetooth._bluetooth as bluez

dev_id = 0    
scanning = True

#Scanner function
def scan():
	try:
		sock = bluez.hci_open_dev(dev_id)

	except Exception as e:
		print("ERROR: Accessing bluetooth device: " + str(e))
		sys.exit(1)

	blescanner = blescan.bleScan()
	blescanner.hci_le_set_scan_parameters(sock)
	blescanner.hci_enable_le_scan(sock)
	
	#Keep scanning until the manager is told to stop.
	while scanning:
		
		returnedDict = blescanner.parse_events(sock, 1)
		print(returnedDict)

#Stop Scanning function
def stop():
	scanning = False


scan()

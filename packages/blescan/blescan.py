# BLE iBeaconScanner based on https://github.com/adamf/BLE/blob/master/ble-scanner.py
# JCS 06/07/14
# BLE scanner based on https://github.com/adamf/BLE/blob/master/ble-scanner.py
# BLE scanner, based on https://code.google.com/p/pybluez/source/browse/trunk/examples/advanced/inquiry-with-rssi.py

# https://github.com/pauloborges/bluez/blob/master/tools/hcitool.c for lescan
# https://kernel.googlesource.com/pub/scm/bluetooth/bluez/+/5.6/lib/hci.h for opcodes
# https://github.com/pauloborges/bluez/blob/master/lib/hci.c#L2782 for functions used by lescan

# performs a simple device inquiry, and returns a list of ble advertizements 
# discovered device

# NOTE: Python's struct.pack() will add padding bytes unless you make the endianness explicit. Little endian
# should be used for BLE. Always start a struct.pack() format string with "<"

import os
import sys
import struct
import bluetooth._bluetooth as bluez
import time


class bleScan:
    def __init__(self):
        self.DEBUG = False
        self.LE_META_EVENT = 0x3e
        self.LE_PUBLIC_ADDRESS = 0x00
        self.LE_RANDOM_ADDRESS = 0x01
        self.LE_SET_SCAN_PARAMETERS_CP_SIZE = 7
        self.OGF_LE_CTL = 0x08
        self.OCF_LE_SET_SCAN_PARAMETERS = 0x000B
        self.OCF_LE_SET_SCAN_ENABLE = 0x000C
        self.OCF_LE_CREATE_CONN = 0x000D

        self.LE_ROLE_MASTER = 0x00
        self.LE_ROLE_SLAVE = 0x01

        # these are actually sub-events of LE_META_EVENT
        self.EVT_LE_CONN_COMPLETE = 0x01
        self.EVT_LE_ADVERTISING_REPORT = 0x02
        self.EVT_LE_CONN_UPDATE_COMPLETE = 0x03
        self.EVT_LE_READ_REMOTE_USED_FEATURES_COMPLETE = 0x04

        # Advertisement event types
        self.ADV_IND = 0x00
        self.ADV_DIRECT_IND = 0x01
        self.ADV_SCAN_IND = 0x02
        self.ADV_NONCONN_IND = 0x03
        self.ADV_SCAN_RSP = 0x04

        # Discovered devices dictionary,
        # this will be updated with scanned devices once it is started
        self.discovered_devices = {}
        self.discovered_devices_buffer = []
        self.discovered_devices_buffer_length = 10

        # Check if parsing is still active
        self.parse_done = False

    def returnnumberpacket(self, pkt):
        myInteger = 0
        multiple = 256
        for c in pkt:
            myInteger += struct.unpack("B", bytes([c]))[0] * multiple
            multiple = 1

        return myInteger

    def returnstringpacket(self, pkt):
        myString = ""
        for c in pkt:
            myString += "%02x" % struct.unpack("B", bytes([c]))[0]

        return myString

    def printpacket(self, pkt):
        for c in pkt:
            sys.stdout.write("%02x " % struct.unpack("B", bytes([c]))[0])

    def get_packed_bdaddr(self, bdaddr_string):
        packable_addr = []
        addr = bdaddr_string.split(':')
        addr.reverse()
        for b in addr:
            packable_addr.append(int(b, 16))

        return struct.pack("<BBBBBB", *packable_addr)

    def packed_bdaddr_to_string(self, bdaddr_packed):

        return ':'.join('%02x'%i for i in struct.unpack("<BBBBBB", bdaddr_packed[::-1]))

    def hci_enable_le_scan(self, sock):
        self.hci_toggle_le_scan(sock, 0x01)

    def hci_disable_le_scan(self, sock):
        self.hci_toggle_le_scan(sock, 0x00)

    def hci_toggle_le_scan(self, sock, enable):
        # hci_le_set_scan_enable(dd, 0x01, filter_dup, 1000);
        # memset(&scan_cp, 0, sizeof(scan_cp));
         #uint8_t         enable;
         #       uint8_t         filter_dup;
        #        scan_cp.enable = enable;
        #        scan_cp.filter_dup = filter_dup;
        #
        #        memset(&rq, 0, sizeof(rq));
        #        rq.ogf = OGF_LE_CTL;
        #        rq.ocf = OCF_LE_SET_SCAN_ENABLE;
        #        rq.cparam = &scan_cp;
        #        rq.clen = LE_SET_SCAN_ENABLE_CP_SIZE;
        #        rq.rparam = &status;
        #        rq.rlen = 1;

        #        if (hci_send_req(dd, &rq, to) < 0)
        #                return -1;
        cmd_pkt = struct.pack("<BB", enable, 0x00)
        bluez.hci_send_cmd(sock, self.OGF_LE_CTL, self.OCF_LE_SET_SCAN_ENABLE, cmd_pkt)

    def hci_le_set_scan_parameters(self, sock):
        old_filter = sock.getsockopt(bluez.SOL_HCI, bluez.HCI_FILTER, 14)
        SCAN_RANDOM = 0x01
        OWN_TYPE = SCAN_RANDOM
        SCAN_TYPE = 0x01

    def parse_events(self, sock, loop_count=100):
        self.parse_done = False

        # Create default device discovery dictionary
        self.discovered_devices = \
            dict(MAC_Address="n/a",
                 UDID="n/a",
                 MAJOR="n/a",
                 MINOR="n/a",
                 TX_Power="n/a",
                 RSSI="n/a")

        old_filter = sock.getsockopt(bluez.SOL_HCI, bluez.HCI_FILTER, 14)

        # perform a device inquiry on bluetooth device #0
        # The inquiry should last 8 * 1.28 = 10.24 seconds
        # before the inquiry is performed, bluez should flush its cache of
        # previously discovered devices
        flt = bluez.hci_filter_new()
        bluez.hci_filter_all_events(flt)
        bluez.hci_filter_set_ptype(flt, bluez.HCI_EVENT_PKT)
        # Set a timeout in order for pasing time out if nothing on socket is received
        sock.settimeout(1)
        # Set socket options
        sock.setsockopt(bluez.SOL_HCI, bluez.HCI_FILTER, flt)

        results = []
        myFullList = []
        for i in range(0, loop_count):
            try:
                # Bocking event, but will time out after the 'settimeout'
                pkt = sock.recv(255)

                ptype, event, plen = struct.unpack("BBB", pkt[:3])
                if event == bluez.EVT_INQUIRY_RESULT_WITH_RSSI:
                    i =0
                elif event == bluez.EVT_NUM_COMP_PKTS:
                    i =0
                elif event == bluez.EVT_DISCONN_COMPLETE:
                    i =0
                elif event == self.LE_META_EVENT:
                    subevent, = struct.unpack("B", bytes([pkt[3]]))
                    pkt = pkt[4:]
                    if subevent == self.EVT_LE_CONN_COMPLETE:
                        pass
                        # le_handle_connection_complete(pkt)
                    elif subevent == self.EVT_LE_ADVERTISING_REPORT:
                        num_reports = struct.unpack("B", bytes([pkt[0]]))[0]
                        report_pkt_offset = 0

                        for k in range(0, num_reports):
                            if self.DEBUG:
                                print("-------------")
                                # print("\tfullpacket: ", printpacket(pkt))
                                print("\tTS:", time.time())
                                print("\tUDID: ", self.printpacket(pkt[report_pkt_offset - 22: report_pkt_offset - 6]))
                                print("\tMAJOR: ", self.printpacket(pkt[report_pkt_offset - 6: report_pkt_offset - 4]))
                                print("\tMINOR: ", self.printpacket(pkt[report_pkt_offset - 4: report_pkt_offset - 2]))
                                print("\tMAC address: ", self.packed_bdaddr_to_string(pkt[report_pkt_offset +
                                                                                          3:report_pkt_offset + 9]))
                                # commented out - don't know what this byte is.  It's NOT TXPower
                                txpower, = struct.unpack("b", bytes([pkt[report_pkt_offset - 2]]))
                                print("\t(Unknown):", txpower)
                                rssi, = struct.unpack("b", bytes([pkt[report_pkt_offset - 1]]))
                                print("\tRSSI:", rssi)

                            # Create a dictionary of discovered devices
                            self.discovered_devices = \
                                dict(TS=time.time(),
                                     MAC_Address=self.packed_bdaddr_to_string(pkt[report_pkt_offset + 3:report_pkt_offset + 9]),
                                     UDID=self.returnstringpacket(pkt[report_pkt_offset - 22: report_pkt_offset - 6]),
                                     MAJOR=self.returnnumberpacket(pkt[report_pkt_offset - 6: report_pkt_offset - 4]),
                                     MINOR=self.returnnumberpacket(pkt[report_pkt_offset - 4: report_pkt_offset - 2]),
                                     TX_Power=struct.unpack("b", bytes([pkt[report_pkt_offset - 2]])),
                                     RSSI=struct.unpack("b", bytes([pkt[report_pkt_offset - 1]])))

                            # Check the length of the buffer,
                            # if it is not yet full, add a new item to the list,
                            # if it reached the defined length then remove the first item and add the new item to the end
                            if len(self.discovered_devices_buffer) != self.discovered_devices_buffer_length:
                                self.discovered_devices_buffer.append(self.discovered_devices)
                            else:
                                self.discovered_devices_buffer.pop(0)
                                self.discovered_devices_buffer.append(self.discovered_devices)

            except Exception as e:
                # Start removing devices from the buffer
                if self.discovered_devices_buffer:
                    self.discovered_devices_buffer.pop(0)

        sock.setsockopt( bluez.SOL_HCI, bluez.HCI_FILTER, old_filter)

        print("\n")
        print(self.discovered_devices_buffer)
        print("\n")

        # Parsing is done
        self.parse_done = True

        return self.parse_done



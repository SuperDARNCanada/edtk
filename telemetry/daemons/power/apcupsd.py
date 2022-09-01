#from __future__ import print_function
import socket
#from collections import OrderedDict

CMD_STATUS = "\x00\x06status".encode()
EOF = "  \n\x00\x00"
SEP = ":"
BUFFER_SIZE = 1024
ALL_UNITS = (
    "Minutes",
    "Seconds",
    "Percent",
    "Volts",
    "Watts",
    "Amps",
    "Hz",
    "C",
    "VA",
    "Percent Load Capacity"
)

def get(host="localhost", port=3551, timeout=30):
    """
    Connect to the APCUPSd NIS and request its status.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((host, port))
    sock.send(CMD_STATUS)
    buffr = ""
    while not buffr.endswith(EOF):
        buffr += sock.recv(BUFFER_SIZE).decode()
    sock.close()
    return buffr


if __name__ == "__main__":
    print(get())

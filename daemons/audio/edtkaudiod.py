# Audio recorder daemon
import pyaudio
import numpy as np
import datetime
import time


config = {'site': 'lab',    # Location of webcam descriptor
          'filepath': '',   # Path to save the audio clips
          'cadence': 60,     # Cadence to take pictures at in seconds
          }
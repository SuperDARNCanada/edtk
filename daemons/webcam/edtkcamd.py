# Webcam daemon
import cv2
import numpy as np
import datetime
from time import sleep

config = {'site': 'lab',
          'camera': 0,
          'filepath': '',
          'rotate': None,
          'scale': None
          }


class Webcam:
    def __init__(self, config):
        self.cam = cv2.VideoCapture(config['camera'])
        self.photo = np.array([])
        self.filename = config['filepath'] + config['site'] + '_cam' + str(config['camera']) + '_'
        self.capture()
        self.save()

    def __del__(self):
        self.cam.release()
        cv2.destroyAllWindows()

    def capture(self):
        _, frame = self.cam.read()
        self.photo = frame

    def save(self):
        t = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.filename += t + '.png'
        cv2.imwrite(self.filename, self.photo)
    #
    # def scale(self):
    #
    # def rotate(self):


if __name__ == '__main__':
    for i in range(5):
        Webcam(config)
        sleep(5)

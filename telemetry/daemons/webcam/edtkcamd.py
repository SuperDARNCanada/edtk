# Webcam daemon
import cv2
import numpy as np
import datetime
import time


config = {'site': 'lab',        # Location of webcam descriptor
          'camera': 0,          # USB camera 0 for first detected, increment for more
          'filepath': '',       # Path to save the images
          'rotate': None,       # Accepts int 90, 180, 270
          'scale': None,        # Percent to scale 100 = 100%
          'mirror': None,       # Axis to flip, 0 = vertical, 1 = horizontal
          'contrast': 0,      # Set the contrast percent from 0 to 100 (0 is Normal)
          'brightness': 255,    # Brightness 0 to 255 (255 is max default)
          'cadence': 60,        # Cadence to take pictures at in seconds
          }


class Webcam:
    def __init__(self, config):
        self.cam = cv2.VideoCapture(config['camera'])
        self.frame = np.array([])
        self.cadence = config['cadence']
        self.filename = config['filepath'] + config['site'] + '_cam' + str(config['camera']) + '_'
        self.rotate = config['rotate']
        self.scale = config['scale']
        self.mirror = config['mirror']
        self.contrast = config['contrast']
        self.brightness = config['brightness']

    def __del__(self):
        self.cam.release()
        cv2.destroyAllWindows()

    def capture(self):
        _, frame = self.cam.read()
        self.frame = frame
        if self.rotate in [90, 180, 270]:
            rot = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
            idx = [90, 180, 270].index(self.rotate)
            self.frame = cv2.rotate(self.frame, rot[idx])
        if self.mirror is [0, 1]:
            self.frame = cv2.flip(self.frame, self.mirror)
        if self.scale is not None:
            width = int(self.frame.shape[1] * self.scale / 100)
            height = int(self.frame.shape[0] * self.scale / 100)
            self.frame = cv2.resize(self.frame, (width, height))
        if (0 < self.contrast <= 100) or (0 <= self.brightness < 255):
            self.brightness = max(0, min(self.brightness, 255))
            self.contrast = max(0, min(self.contrast, 100))
            contrast = self.brightness * self.contrast / 100
            self.frame = cv2.normalize(self.frame, None, alpha=contrast, beta=self.brightness,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    def save(self):
        caption = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        file_time = caption.replace(" ", "").replace(":", "").replace("-", "")
        cv2.putText(self.frame, caption, (0, 24), cv2.FONT_HERSHEY_PLAIN, 2, 255)
        cv2.imwrite(self.filename + file_time + '.png', self.frame)

    def take_photo(self):
        self.capture()
        # cv2.imshow(self.filename, self.frame)
        self.save()

    def video(self):
        while True:
            self.capture()
            cv2.imshow(self.filename, self.frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    cam = Webcam(config)
    # cam.take_photo()
    # exit()

    # # schedule.every(cam.cadence).seconds.do(cam.take_photo)
    # time.sleep(60.0 - time.localtime().tm_sec)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)

    time.sleep(60.0 - time.localtime().tm_sec)
    while True:
        cam.take_photo()
        cv2.waitKey(1)
        time.sleep(cam.cadence)

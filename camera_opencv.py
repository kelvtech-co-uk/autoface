import os
import cv2
from base_camera import BaseCamera


class Camera(BaseCamera):
    #video_source = 0
    
    # Hardcode local network URL for webcam go2rtc feed
    video_source = "rtsp://192.168.1.99:8554/blackprostation"

    # def __init__(self):
    #     if os.environ.get('OPENCV_CAMERA_SOURCE'):
    #         Camera.set_video_source(os.environ['OPENCV_CAMERA_SOURCE'])
    #     super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
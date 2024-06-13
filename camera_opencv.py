import os
import cv2
from base_camera import BaseCamera
#from retinaface import RetinaFace

class Camera(BaseCamera):
    #video_source = 0
    
    # Hardcode local network URL for webcam go2rtc feed
    video_source = "rtsp://192.168.1.99:8554/blackprostation"

    # def __init__(self):
    #     if os.environ.get('OPENCV_CAMERA_SOURCE'):
    #         Camera.set_video_source(os.environ['OPENCV_CAMERA_SOURCE'])
    #     super(Camera, self).__init__()

    # Not sure what the next 3 lines are for as I cant find 'source' anywhere
    # @staticmethod
    # def set_video_source(source):
    #     Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        
        # Load the cascade
        #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()
            
            #faces = RetinaFace.detect_faces(img)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            text = ','.join(str(v) for v in faces)
            cv2.putText(img, text, (1, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
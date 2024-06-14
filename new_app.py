#!/usr/bin/env python
from flask import Flask, render_template, Response
# import threading
# import datetime
# import time
import cv2

# outputFrame = None
# lock = threading.Lock()
app = Flask(__name__, template_folder='.')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
video_source = "rtsp://192.168.1.99:8554/blackprostation"
vs = cv2.VideoCapture(video_source)

def genframes():    
    while True:
        # read current frame
        _, img = vs.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        text = ','.join(str(v) for v in faces)
        cv2.putText(img, text, (1, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # encode as a jpeg image and return it
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', img)[1].tobytes() + b'\r\n--frame\r\n'

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(genframes(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=False, use_reloader=False)

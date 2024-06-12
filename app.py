#!/usr/bin/env python
from importlib import import_module
import os
from flask import Flask, render_template, Response

# # Check environment variable
# if os.environ.get('CAMERA'):
#     Camera = import_module('camera_' + os.environ['CAMERA']).Camera
# else:
#     from camera import Camera

# Bypass checking for environment variable and hardcode opencv
Camera = import_module('camera_opencv').Camera

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    # index.html needs to be in template\index.html or alter the Flash app definition
    return render_template('index.html') 

def gen(camera):
    """Video streaming generator function."""
    yield b'--frame\r\n'
    while True:
        frame = camera.get_frame()
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
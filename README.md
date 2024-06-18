# autoface
Real-time camera monitoring, face detection and recognition inside a docker container using OpenCV.

## Code Journey
* archived_code/v1:   Hack of Miguel Grinberg's and PORNPASOK SOOKYEN's code pulling frames off my webcam using go2rtc and streaming to a browser session using Flask
* archived_code/v2:   Face detection using Haar cascades from OpenCV
* archived_code/v3:   Face detection using YuNet with options to pass in either a still image or a video with a matching output including detection visualisation
* current branch:     Face detection and recognition using YuNet and SFace with video output including basic recognition visualisation

## To Do
The ever chaning goal posts!
- [x] Implement sface recognition :white_check_mark:
- [ ] Add more recognition visualisation data to the output video
- [ ] Optimise resource impacts ensuring OpenCL usage.  Review OpenVINO for any tangible gain.
- [ ] Parse detection and recognition outputs into more human readable text
- [ ] Implementing appropriate Threading
- [ ] Implement Collections/Persons lookup in the recognition process
- [ ] Implement API and/or MQTT wrapper considering integration with the Home Assistant and Node Red projects

My dev environment uses a USB Logitech Brio webcam on my workstation streamed into my container host/container using a great project credited below called go2rtc.  I'm very much looking to keep this small and lightweight and will optimise the processing overheads as best I can.  My end-state usecase is to process either jpeg snapshots or an rtsp feed from my Unifi G4 Doorbell camera for person and face detection and ultimately recogition.  I want to integrate this with Home Assistant and/or Node-Red to onward process the recognition output into events and alerts with my vairous IoT devices.

### Fair warning
I'm very much learning and have little experience with opencv, python & coding in general.  Any advice e.g. "read up on this..." or "check this code/link out..." is welcomed!

### Credits (great projects, code and general inspiration)
* https://github.com/AlexxIT/go2rtc
* https://github.com/miguelgrinberg/flask-video-streaming
* https://github.com/pornpasok/opencv-stream-video-to-web/blob/main/webstreaming.py
* https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
* https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
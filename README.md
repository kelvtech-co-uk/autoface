# autoface
Real-time camera monitoring, face detection and recognition inside a docker container using OpenCV.

### Code Journey
* **archived_code/v1:**   Hack of Miguel Grinberg's and PORNPASOK SOOKYEN's code pulling frames off my webcam using go2rtc and streaming to a browser session using Flask
* **archived_code/v2:**   Face detection using Haar cascades from OpenCV
* **archived_code/v3:**   Face detection using YuNet with options to pass in either a still image or a video with a matching output including detection visualisation
* **sface_recognition branch:**     Face detection and recognition using YuNet and SFace with video output including ~~basic~~ recognition visualisation. Code also runs either in Windows with windowed real-time output or in a docker container with saved .mp4 video output.

### To Do
The ever chaning goal posts!
- [x] Implement sface recognition :beer:
- [x] Add more recognition visualisation data to the output video :beers:
- [ ] Parse detection and recognition outputs into more human readable text
- [ ] Implementing appropriate Threading
- [ ] Implement Collections/Persons lookup in the recognition process
- [ ] Optimise resource impacts ensuring OpenCL usage.  Review OpenVINO for any tangible gain.
- [ ] Implement API and/or MQTT wrapper considering integration with the Home Assistant and Node Red projects


My dev environment uses a USB Logitech Brio webcam on my workstation streamed into my container host/container using a great project credited below called go2rtc.  I'm very much looking to keep this small and lightweight and will optimise the processing overheads as best I can.  My end-state usecase is to process either jpeg snapshots or an rtsp feed from my Unifi G4 Doorbell camera for person and face detection and ultimately recogition.  I want to integrate this with Home Assistant and/or Node-Red to onward process the recognition output into events and alerts with my vairous IoT devices.

### Observations
My container host is an unraid server with a Intel Core i5-14500.  I have a USB connected PSU where unraid is polling the bus and reporting various stats including current power draw.  The table below is an early data capture of the code run cost.
| Power (watts) | Scenario |
| --- | --- |
| 76w | Container host baseline |
| 86w (+10w) | Running code but with no detections/recognitions |
| 103w (+27w-c) | 1 face detected |
| 119w (+43w-c) | 2 faces detected |

### Fair warning
I'm very much learning and have little experience with opencv, python & coding in general.  Any advice e.g. "read up on this..." or "check this code/link out..." is welcomed!

### Credits (great projects, code and general inspiration)
* https://github.com/AlexxIT/go2rtc
* https://github.com/miguelgrinberg/flask-video-streaming
* https://github.com/pornpasok/opencv-stream-video-to-web
* https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
* https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
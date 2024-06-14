# autoface
Real-time camera monitoring, face detection and recognition inside a docker container using OpenCV.

Face detection currently using Haar cascades from OpenCV but am moving to YuNet or Retinaface in near future.

My dev environment uses a USB Logitech Brio webcam on my workstation streamed into my container host/container using a great project credited below called go2rtc.  I'm very much looking to keep this small and lightweight and will optimise the processing overheads as best I can.  My end-state usecase is to process either jpeg snapshots or an rtsp feed from my Unifi G4 Doorbell camera for person and face detection and ultimately recogition.  I want to the integrate with Home Assistant and/or Node-Red to onward process the recognition output into events and alerts with my vairous IoT devices.

### Fair warning
I'm very much learning and have little experience with opencv, python & coding in general.  Any advice e.g. "read up on this..." or "check this code/link out..." is welcomed!

### Credits (great projects, code and general inspiration)
* https://github.com/AlexxIT/go2rtc
* https://github.com/miguelgrinberg/flask-video-streaming
* https://github.com/pornpasok/opencv-stream-video-to-web/blob/main/webstreaming.py

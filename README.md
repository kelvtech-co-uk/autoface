# autoface
A self-hosted, quick, light weight and real-time camera monitoring application which includes face detection and recognition using OpenCV.

### Code Journey
* **archived_code/v1:**   Hack of Miguel Grinberg's and PORNPASOK SOOKYEN's code pulling frames off my webcam using go2rtc and streaming to a browser session using Flask
* **archived_code/v2:**   Face detection using Haar cascades from OpenCV
* **archived_code/v3:**   Face detection using YuNet with options to pass in either a still image or a video with a matching output including detection visualisation
* **sface_recognition branch:**     Face detection and recognition using YuNet and SFace with video output including ~~basic~~ recognition visualisation. Code also runs either in Windows with windowed real-time output or in a docker container with saved .mp4 video output.
* **main branch:**  Merged above branch back into main and am staying here for a little while.

### To Do
The ever chaning goal posts!
- [x] Implement sface recognition :beer:
- [x] Add more recognition visualisation data to the output video :beers:
- [x] Implement checks and corrections to target image size to optimise for YuNet detection model
- [x] Implement code to look up and filter target image files stored in a collation/person folder structure
- [x] Update the detection code to use the collation/person/target file list
- [ ] Make code more readable with useful comments and by using functions and/or seperate .py files as needed
- [ ] Consider better way to integrate opencv_zoo repo to access YuNet and Sface dynamically rather than keeping static files seperately
- [x] Parse detection and recognition outputs into more human readable text
- [ ] Implementing appropriate Threading and/or Multiprocessing.
- [ ] Review benefits case of pure OpenCL given early observations on gains/costs
- [ ] Short validation tests and benefits case review of using the OpenVINO (cv.dnn.DNN_BACKEND_INFERENCE_ENGINE) backend with an appropriately prepared docker image
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

#### OPENCL
Using the targetId=cv.dnn.DNN_TARGET_OPENCL for both detector and recogniser does successfully move workloads onto the GPU but testing on my container host showed the power draw was largely the same with OPENCL possibly drawing more overall by ~5 watts.  CPU utilisation however was clearly improved when using OPENCL.  On my Windows workbench (AMD w/ Nvidia GPU) whilst OPENCL functioned it was clearly not optimised as GPU utilisation reported near 100% via the Task Manager and the frame rate in the windowed output was 1-5 FPS.

Lastly load and execution time of the programmed was different; measured on my container host it took 12.5 seconds for the output video to start to be written when using OPENCL vs. 2 seconds when using just CPU.  

**_UPDATE_** Enter the world of OpenCL Kernel tuning, setting an environment variable called OPENCV_OCL4DNN_CONFIG_PATH to a permanent directory and starting the container causes Opencl to start auto-tuning against my specific hardware.  The directory specificied in the environment variable gets populated with tuning configuration data.  Running the container post autotuning reduces the initial program load & execution time by 50% in my testing!

## Fair warning
This is an "I'm learning Python, OpenCV and some associated tools" project, I'm certain that all of what I've done/am doing/plan to-do has already been done elsewhere and to a significantly better standard.  Check out some of the repos credited below rather than take my code ;-)

### Credits (great projects, code and general inspiration)
* https://github.com/AlexxIT/go2rtc
* https://github.com/miguelgrinberg/flask-video-streaming
* https://github.com/pornpasok/opencv-stream-video-to-web
* https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
* https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
* https://github.com/ShiqiYu/libfacedetection
* https://github.com/serengil/deepface
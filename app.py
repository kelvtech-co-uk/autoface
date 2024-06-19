import numpy as np
import cv2 as cv
# import argparse
import sys
import platform
import datetime
import time
from yunet import YuNet
from sface import SFace

# parser = argparse.ArgumentParser()
# parser.add_argument("--input", help='still or video')
# args = parser.parse_args()

# Ensure OpenCL is used
cv.ocl.setUseOpenCL(True)

def get_os():
    system = platform.system()
    if system == "Windows":
        return "Windows"
    elif system == "Linux":
        return "Linux"
    else:
        return "Unknown"
    
print('Running in', get_os())

def draw_text(query_image, text, pos=(0, 0), font=cv.FONT_HERSHEY_SIMPLEX, font_scale=0.5, text_color=(255, 255, 255), font_thickness=1, text_color_bg=(0, 0, 0), border=1):
    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(query_image, (x, y + border), (x + text_w, y - text_h - border), text_color_bg, -1)
    cv.putText(query_image, text, (x, y), font, font_scale, text_color, font_thickness)

    return #text_size

def visualize(query_image, query_faces, matches, scores, fps, detection_time, inference_time): # target_size: (h, w)
    matched_box_color = (0, 255, 0)    # BGR
    mismatched_box_color = (0, 0, 255) # BGR
        
    # Validate results
    assert query_faces.shape[0] == len(matches), "number of query_faces needs to match matches"
    assert len(matches) == len(scores), "number of matches needs to match number of scores"
    
    draw_text(query_image, text=f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", pos=(0, 30))
    #cv.putText(query_image,f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    if fps is not None:
        draw_text(query_image, text=f"FPS: {fps:.0f}", pos=(0, 45))
        #cv.putText(query_image, 'FPS: {:.2f}'.format(fps), (0, 45), cv.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    # Only report detection time if there are faces found in the frame:
    if query_faces.shape[0] > 0:
        draw_text(query_image, text=f"Detection: {detection_time:.4f}ms", pos=(0, 60))
        #cv.putText(query_image,f"Detection: {detection_time:.4f}ms", (0, 60), cv.FONT_HERSHEY_SIMPLEX, 0.4, text_color)    

    # Draw bbox
    for index, match in enumerate(matches):
        bbox = query_faces[index][:4]
        x, y, w, h = bbox.astype(np.int32)
        box_color = matched_box_color if match else mismatched_box_color
        cv.rectangle(query_image, (x, y), (x + w, y + h), box_color, 2)

        score = scores[index]
        text_color = matched_box_color if match else mismatched_box_color
        draw_text(query_image, text=f"Inference: {inference_time:.4f}ms", pos=(0, 75))
        cv.putText(query_image, "{:.2f}".format(score), (x, y - 5), cv.FONT_HERSHEY_DUPLEX, 0.4, text_color)

    return query_image

# Instantiate YuNet & SFace
detector = YuNet(modelPath='face_detection_yunet_2023mar.onnx',
                inputSize=[960, 720], 
                confThreshold=0.9,    #Usage: Set the minimum needed confidence for the model to identify a face. Smaller values may result in faster detection, but will limit accuracy.
                nmsThreshold=0.3,     #Usage: Suppress bounding boxes of iou >= nms_threshold.
                topK=5000,            #Usage: Keep top_k bounding boxes before NMS.
                backendId=cv.dnn.DNN_BACKEND_OPENCV,
                targetId=cv.dnn.DNN_TARGET_CPU)

recognizer = SFace(modelPath='face_recognition_sface_2021dec.onnx',
                    disType=0,      #Usage: Distance type. \'0\': cosine, \'1\': norm_l1. Defaults to \'0\'
                    backendId=cv.dnn.DNN_BACKEND_OPENCV,
                    targetId=cv.dnn.DNN_TARGET_CPU)

# Load target image
target_image = cv.imread("target2.jpeg")
target_image = cv.resize(target_image, (180,320), interpolation=cv.INTER_LINEAR)

# Detect faces in target
detector.setInputSize([target_image.shape[1], target_image.shape[0]])
target_face = detector.infer(target_image)
if target_face.shape[0] == 0:
    sys.exit("No faces deteceted in query source")

# Load query video source
query_source_url = "rtsp://192.168.1.99:8554/blackprostation"
#query_source_url = "rtsps://192.168.1.1:7441/EOEohGh0eoXIWf28"
#query_source_url = "event.mp4"

query_source = cv.VideoCapture(query_source_url)
query_source_width = int(query_source.get(cv.CAP_PROP_FRAME_WIDTH))
query_source_height = int(query_source.get(cv.CAP_PROP_FRAME_HEIGHT))

# ONLY FOR MY DEV ENV
if get_os() == "Linux":
    output = cv.VideoWriter('/mnt/cache/processing/result.mp4', cv.VideoWriter_fourcc(*'mp4v'), 25, (query_source_width, query_source_height), True)
elif get_os() == "Windows":
    output = cv.VideoWriter('result.mp4', cv.VideoWriter_fourcc(*'mp4v'), 25, (query_source_width, query_source_height), True)

fps = cv.TickMeter()
frame_skip = 2
try:
    while True:
        # read a frame from the query video source
        fps.start()
        hasFrame, query_image = query_source.read()
        if not hasFrame:
            sys.exit("Unable to grab a frame from query_source")

        # Attempt to detect faces
        det_start_time = time.time()
        detector.setInputSize([query_image.shape[1], query_image.shape[0]])
        query_faces = detector.infer(query_image)
        det_end_time = time.time()
        
        detection_time = det_end_time - det_start_time

        # Match faces to target
        scores = []
        matches = []
        infer_start_time = time.time()
        for face in query_faces:
            result = recognizer.match(target_image, target_face[0][:-1], query_image, face[:-1])
            scores.append(result[0])
            matches.append(result[1])
        
        infer_end_time = time.time()
            
        inference_time = infer_end_time - infer_start_time
        
        fps.stop()
        
        # Draw results
        image = visualize(query_image, query_faces, matches, scores, fps.getFPS(), detection_time, inference_time)
        
        if get_os() == "Linux":
            output.write(image)
        elif get_os() == "Windows":
            cv.imshow('autoface', image)
            if cv.waitKey(1) == ord('q'):
                break
        
        for _ in range(frame_skip - 1):
            query_source.read()

except KeyboardInterrupt:
    pass

if get_os() == "Linux":
    output.release()
elif get_os() == "Windows":
    cv.destroyAllWindows()

query_source.release()
print('\nProgram exited')





